import socket
import numpy as np
import json

# =========================================================
# [STEP 0] 캘리브레이션 행렬 설정 (카메라 → 로봇)
# =========================================================
# 3x4 [ R | t ] 형식의 행렬 (단위: mm)
TRANSFORMATION_MATRIX = np.array([
    [0.01665455, 0.97822465, 0.02832482, 377.40874423],
    [0.99141691, -0.02245248, -0.00914220,   8.52529268],
    [-0.01228417, 0.00438147, -0.97705115, 390.87641972],
], dtype=float)

# 회전/이동 분리
R_CAM2ROB = TRANSFORMATION_MATRIX[:, :3]   # (3x3) 회전
T_CAM2ROB = TRANSFORMATION_MATRIX[:, 3]    # (3,)   평행이동

# 서버 설정 (DRL 코드에 맞춤)
SERVER_HOST = "0.0.0.0"     # 모든 인터페이스에서 수신
SERVER_PORT = 200           # DRL에서 200번 포트 사용 중
JSONL_PATH = "face_path_points_10mm.jsonl"  # 점/법선이 저장된 파일 경로


# =========================================================
# [STEP 1] 카메라 → 로봇 좌표 변환 함수들
# =========================================================
def cam_point_to_robot(point_cam_m: np.ndarray) -> np.ndarray:
    """
    카메라 좌표계 점 (m 단위) → 로봇 좌표계 점 (mm 단위)
    point_cam_m : (3,)  [m]
    return      : (3,)  [mm]
    """
    # 1) m → mm
    p_cam_mm = point_cam_m * 1000.0

    # 2) 로봇 좌표계로 변환: p_robot = R * p_cam + t
    p_robot_mm = R_CAM2ROB @ p_cam_mm + T_CAM2ROB
    return p_robot_mm


def cam_normal_to_robot(normal_cam: np.ndarray) -> np.ndarray:
    """
    카메라 좌표계 법선 벡터 → 로봇 좌표계 법선 벡터
    (법선은 방향만 중요하므로 회전만 적용, 평행이동은 무시)
    """
    n = R_CAM2ROB @ normal_cam
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        # 이상한 값일 경우 기본값 Z+ 사용
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return n / norm


# =========================================================
# [STEP 2] 법선 → 회전행렬 → Euler ZYZ (deg) 만들기
#   - posx([...]) 기본 ori_type = Euler ZYZ, deg
# =========================================================
def frame_from_normal_z(normal_robot: np.ndarray,
                        world_up: np.ndarray = np.array([0.0, 0.0, 1.0])
                        ) -> np.ndarray:
    """
    normal_robot : (3,) 로봇 좌표계 기준 법선 (단위벡터)
    world_up     : (3,) 기준 up 벡터 (보통 [0,0,1])
    return       : (3,3) 회전행렬 R (열: x, y, z 축)
    - TCP의 Z축을 법선 방향으로 맞추는 기준 좌표계를 만든다.
    """
    z = normal_robot / (np.linalg.norm(normal_robot) + 1e-9)  # TCP Z축

    # z와 거의 평행하지 않은 기준 벡터 선택
    ref = world_up
    if abs(np.dot(z, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    x = np.cross(ref, z)
    x = x / (np.linalg.norm(x) + 1e-9)

    y = np.cross(z, x)

    # 각 열이 x, y, z 축이 되도록 구성
    R = np.column_stack((x, y, z))
    return R


def euler_zyz_from_R(R: np.ndarray):
    """
    회전행렬 R → Euler ZYZ (rad)
    R = Rz(alpha) * Ry(beta) * Rz(gamma)
    """
    # beta
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))

    if abs(np.sin(beta)) < 1e-9:
        # 특이점: beta ≈ 0 or pi
        alpha = np.arctan2(R[1, 0], R[0, 0])
        gamma = 0.0
    else:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])

    return alpha, beta, gamma


def euler_zyz_deg_from_R(R: np.ndarray) -> np.ndarray:
    """
    회전행렬 R → Euler ZYZ (deg)
    return: np.array([A_deg, B_deg, C_deg])
    """
    alpha, beta, gamma = euler_zyz_from_R(R)
    return np.array([
        np.degrees(alpha),
        np.degrees(beta),
        np.degrees(gamma),
    ], dtype=float)


# =========================================================
# [STEP 3] JSONL 한 줄(record) → [x,y,z,rx,ry,rz] 변환
# =========================================================
def pose_from_record(record: dict) -> np.ndarray:
    """
    JSONL 한 줄에서 x,y,z,rx,ry,rz 생성
    - record 안에 다음 key들이 있다고 가정:
      X_m, Y_m, Z_m, nx, ny, nz
    """
    # 1) 카메라 좌표계 점/법선 읽기
    p_cam_m = np.array([
        float(record["X_m"]),
        float(record["Y_m"]),
        float(record["Z_m"]),
    ], dtype=float)

    n_cam = np.array([
        float(record["nx"]),
        float(record["ny"]),
        float(record["nz"]),
    ], dtype=float)

    # 2) 카메라 → 로봇 변환
    p_robot_mm = cam_point_to_robot(p_cam_m)   # (3,) [mm]
    n_robot = -cam_normal_to_robot(n_cam)       # (3,) 단위벡터

    # 3) 법선 → 회전행렬 → Euler ZYZ(deg)
    R_tool = frame_from_normal_z(n_robot)
    A_deg, B_deg, C_deg = euler_zyz_deg_from_R(R_tool)

    x_mm, y_mm, z_mm = p_robot_mm

    # rx,ry,rz 자리에 Euler ZYZ(deg)를 그대로 넣어줌
    pose = np.array([x_mm, y_mm, z_mm, A_deg, B_deg, C_deg], dtype=float)
    return pose


# =========================================================
# [STEP 4] JSONL 전부 읽어서 pose 리스트 만들기
# =========================================================
def load_all_poses(jsonl_path: str):
    """
    JSONL 파일 전체를 읽어서
    [ [x,y,z,rx,ry,rz], ... ] 리스트로 반환
    """
    poses = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pose = pose_from_record(record)
            poses.append(pose)
    print(f"[INFO] 로드된 포인트 개수: {len(poses)}")
    return poses


# =========================================================
# [STEP 5] 소켓 서버: DRL에서 'shot' 요청마다 다음 포즈 전송
# =========================================================
def start_server():
    print(f"[INFO] 서버 시작: {SERVER_HOST}:{SERVER_PORT}")
    print(f"[INFO] JSONL 파일: {JSONL_PATH}")

    # 미리 전체 포즈 로딩 (한 번만)
    poses = load_all_poses(JSONL_PATH)
    if len(poses) == 0:
        print("[ERROR] JSONL에 데이터가 없습니다.")
        return

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(1)

    while True:
        print("[INFO] 클라이언트 접속 대기 중...")
        conn, addr = server.accept()
        print(f"[INFO] 연결됨: {addr}")

        # 새 클라이언트마다 처음 포인트부터 시작
        idx = 0

        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("[INFO] 클라이언트 연결 종료")
                    break

                cmd = data.decode().strip().lower()
                print(f"[RECV] {cmd}")

                # DRL에서 "shot\r\n" 보냄
                if cmd == "shot":
                    pose = poses[idx]

                    # "x,y,z,rx,ry,rz\r\n" 형식으로 전송
                    msg = ",".join(f"{v:.6f}" for v in pose) + "\r\n"
                    conn.sendall(msg.encode())
                    print(f"[SEND] {msg.strip()}")

                    # 마지막 포인트까지 가면, 계속 마지막 포인트 유지
                    if idx < len(poses) - 1:
                        idx += 1

                elif cmd in ("quit", "exit", "close"):
                    print("[INFO] 클라이언트 요청으로 연결 종료")
                    conn.sendall("BYE\r\n".encode())
                    break

                else:
                    # 모르는 명령은 무시하거나 로그만 찍기
                    print(f"[WARN] 알 수 없는 명령: {cmd}")
                    # 필요하면 에러 메시지도 보낼 수 있음
                    # conn.sendall("UNKNOWN_CMD\r\n".encode())
        except Exception as e:
            print(f"[CONNECTION ERROR] {e}")
        finally:
            conn.close()
            print("[INFO] 클라이언트 소켓 닫힘")

    # (무한 루프라 여기까지는 보통 안 옴)
    server.close()


# =========================================================
# [STEP 6] 메인
# =========================================================
if __name__ == "__main__":
    start_server()
