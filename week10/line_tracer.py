# line_tracer_unified.py (반사광 제거 개선 버전)
# 목표: 원본 크기를 유지하면서 흰색 라인과 노란색 라인 마스크를 분리해 출력한다.
# 튜닝 포인트: 노란색 라인 검출 시 채도 하한(S_min)을 높여 바닥의 '흰색 반사광'을 제거!

import cv2 # OpenCV 라이브러리: 이미지 처리의 핵심!
import numpy as np # NumPy 라이브러리: 배열(행렬) 계산을 효율적으로 처리 (이미지 데이터는 큰 배열이니까!)
import os # OS 모듈: 파일 경로와 폴더 관리를 위해 사용
import glob # Glob 모듈: 특정 패턴의 파일 목록을 가져올 때 사용 (예: .jpg 파일 전부)
import sys # System 모듈: 파이썬 실행 환경 관련 정보 (자동 실행 모드를 위해 사용)

# =========================================================
# 👇 🌟 실행 환경에 맞춰 이 경로를 반드시 수정하세요 🌟 👇
# =========================================================
# 이 폴더 안에 네가 처리할 이미지 원본들이 있어야 해
DEFAULT_INPUT_DIR = '/home/minjun/minjun-file'
# 결과물이 저장될 최상위 폴더
DEFAULT_OUTPUT_BASE_DIR = './out_results' 
# =========================================================

# --- 공통 파라미터 ---
COMMON_PARAMS = {
    'width': 640, # 이미지 처리 속도를 위해 리사이즈할 너비. 높이는 비율에 맞춰 자동 조정돼.
    'height': None, # 높이는 비율 유지
    # ROI(관심 영역) 설정: 상단 40%는 제외하고 하단 60% 영역만 처리
    'roi_bottom_ratio': 0.6, 
    # ROI 상단에서 좌우를 각각 100픽셀씩 줄여서 사다리꼴 모양을 만듦 (시야각처럼)
    'roi_width_reduction': 100, 
    
    # 모폴로지 연산 커널: 라인 주변의 작은 노이즈 제거 및 끊긴 라인 연결에 사용
    'k_open': (3,3), # OPEN 연산 (침식 -> 팽창): 작은 노이즈(점) 제거. 커널이 작을수록 라인 침식이 덜해.
    'k_close': (7,7), # CLOSE 연산 (팽창 -> 침식): 끊긴 라인을 연결하고 구멍을 메움.
    
    # Hough Line Transform 파라미터: 검출할 직선의 최소 길이와 최대 간격
    'min_line_length': 30, # 이 길이보다 짧은 직선은 무시해 (노이즈 방지)
    'max_line_gap': 30, # 이 간격 이내로 떨어진 라인들은 하나로 연결하려고 시도해
    # Canny Edge Detection 임계값: Hough 변환 전에 엣지(윤곽선)를 검출하는 기준
    'canny_threshold1': 50, 
    'canny_threshold2': 150, 
    'keep_only_lines': False # 결과 이미지에 원본을 덧칠할지 (False) 검은 배경만 남길지 (True)
}

# --- 흰색 라인만 추출 파라미터 ---
WHITE_LINE_PARAMS = {
    **COMMON_PARAMS, # 공통 파라미터를 그대로 가져와서 사용
    'output_subdir': 'white_lines', # 흰색 결과는 이 폴더에 저장
    # 흰색 검출 범위 (BGR이 아닌 HSV 색 공간으로 설정)
    'white_lower': np.array([0, 0, 180], dtype=np.uint8), # H(색조), S(채도), V(명도)의 최솟값
    'white_upper': np.array([180, 25, 255], dtype=np.uint8), # H, S, V의 최댓값
    # V_min을 180으로 설정해 어두운 회색은 흰색으로 검출되지 않도록 막고 있어.
    # S_max를 25로 낮게 설정해 채도가 조금이라도 있는 색(노란색, 빨간색)이 흰색으로 오인되지 않도록 해.
    
    # 이 아래의 노란색 파라미터는 흰색 마스크를 만들 때는 사용되지 않지만, 함수의 인자로 필요해서 남겨둬.
    'yellow_lower': np.array([15, 80, 80], dtype=np.uint8), 
    'yellow_upper': np.array([40, 255, 255], dtype=np.uint8), 
}

# --- 노란색 라인만 추출 파라미터 ---
YELLOW_LINE_PARAMS = {
    **COMMON_PARAMS, 
    'output_subdir': 'yellow_lines', 
    
    # 🌟🌟 튜닝 포인트: 채도(S) 하한을 높여서 채도가 낮은 반사광 배제 🌟🌟
    # HSV: [H_min, S_min, V_min]
    'yellow_lower': np.array([15, 120, 80], dtype=np.uint8), # S_min을 120으로 올려서 채도가 낮은 '흰색 반사광'을 노란색으로 인식하지 않게 해!
    # H 범위: 15~35 (노란색 색조)
    'yellow_upper': np.array([35, 255, 255], dtype=np.uint8),
    
    # 이 아래의 흰색 파라미터는 노란색 마스크를 만들 때는 사용되지 않음.
    'white_lower': np.array([0, 0, 200], dtype=np.uint8), 
    'white_upper': np.array([180, 60, 255], dtype=np.uint8), 
}
# =========================================================


# 폴더가 없으면 만들어주는 함수 (파일 저장을 위해 필수!)
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# 이미지를 리사이즈하여 처리 속도를 높이고, 최종 결과물에 사용할 원본 크기를 맞추는 함수
def resize_image(img, target_width=None, target_height=None):
    h, w = img.shape[:2]
    # 너비(width) 기준으로 비율을 유지하며 리사이즈
    if target_width:
        r = target_width / float(w)
        return cv2.resize(img, (target_width, int(h * r)), interpolation=cv2.INTER_AREA)
    else:
        return img

# ROI(관심 영역) 마스크를 생성하고 적용하는 함수
# 사다리꼴 모양으로 마스크를 씌워 불필요한 상단 영역(예: 천장)의 노이즈를 제거해.
def apply_roi_mask(img_mask, params):
    h, w = img_mask.shape[:2]
    
    # ROI 시작 y 좌표 계산 (예: 0.6이면 아래쪽 60%부터 시작)
    top_y = int(h * (1 - params['roi_bottom_ratio']))
    
    # 사다리꼴 꼭짓점: (x, y) 순서
    vertices = np.array([
        [(0, h), # 좌측 하단 (전체 너비)
         (w // 2 - params['roi_width_reduction'], top_y), # 좌측 상단 (중앙으로 좁아짐)
         (w // 2 + params['roi_width_reduction'], top_y), # 우측 상단 (중앙으로 좁아짐)
         (w, h)] # 우측 하단 (전체 너비)
    ], dtype=np.int32)
    
    mask_roi = np.zeros_like(img_mask)
    cv2.fillPoly(mask_roi, vertices, 255) # 사다리꼴 영역만 흰색(255)으로 채움
    
    return cv2.bitwise_and(img_mask, mask_roi) # 색상 마스크와 ROI 마스크를 합쳐서 최종 마스크를 만듦

# HSV 색 공간을 이용해 원하는 색상(라인)만 추출하는 함수
def color_mask_hsv(img_bgr, yellow_lower, yellow_upper, white_lower, white_upper, mode='white'):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) # BGR -> HSV로 변환 (색상 구분이 더 쉬움)
    if mode == 'white':
        mask_white = cv2.inRange(hsv, white_lower, white_upper) # 흰색 범위에 있는 픽셀만 마스크(흰색)로 표시
        return mask_white
    elif mode == 'yellow':
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper) # 노란색 범위에 있는 픽셀만 마스크(흰색)로 표시
        return mask_yellow
    else:
        return None # 현재 사용하지 않는 모드

# 모폴로지 연산과 가우시안 블러를 이용해 마스크를 다듬는 함수
def refine_mask(mask, kernel_open, kernel_close):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_close)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open) # 잡티 제거
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close) # 라인 연결
    m = cv2.GaussianBlur(m, (5,5), 0) # 경계선 부드럽게 처리
    _, m = cv2.threshold(m, 50, 255, cv2.THRESH_BINARY) # 다시 이진화 (선명한 흑백 마스크)
    return m

# Hough Line Transform을 이용해 최종 라인 검출 (현재는 마스크만 출력하므로 주석 처리된 상태로 두자)
def detect_lines_by_hough(mask, min_line_length, max_line_gap, canny_threshold1, canny_threshold2):
    edges = cv2.Canny(mask, canny_threshold1, canny_threshold2, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

# 라인과 윤곽선을 원본 이미지에 덧그리는 함수 (현재는 마스크만 저장하므로 사용 안 함)
def draw_results(original, mask, lines, keep_only_lines=False):
    # 이 부분은 lined.png를 만들 때 필요하므로, 마스크만 저장할 때는 호출되지 않아요.
    pass

# 하나의 이미지 파일을 처리하고 마스크를 저장하는 핵심 함수
def process_image_file(path, output_base_dir, params, mode):
    img = cv2.imread(path)
    if img is None:
        print("Error: Can't read", path)
        return
        
    original_for_draw = resize_image(img.copy(), params['width'], params['height']) # 원본 크기 유지용
    process_img = original_for_draw.copy() # 실제 처리에 사용할 리사이즈된 이미지
    
    # 1. HSV 색상 마스크 생성
    color_filtered_mask = color_mask_hsv(process_img, 
                                         params['yellow_lower'], params['yellow_upper'],
                                         params['white_lower'], params['white_upper'], mode=mode)
    
    # 2. ROI 마스크 적용 (관심 영역만 남김)
    roi_masked_color_mask = apply_roi_mask(color_filtered_mask, params)

    # 3. 마스크 다듬기 (노이즈 제거 및 라인 연결)
    refined_mask = refine_mask(roi_masked_color_mask, params['k_open'], params['k_close'])
    
    # 최종 파일 저장 경로 설정
    output_dir = os.path.join(output_base_dir, params['output_subdir'])
    ensure_dir(output_dir)
    name = os.path.splitext(os.path.basename(path))[0]
    mask_outp = os.path.join(output_dir, f"{name}_mask.png")
    
    # 🌟 4. 마스크 저장: 노란색은 노란색으로, 흰색은 흑백으로 저장 🌟
    if mode == 'yellow':
        # 흑백 마스크를 BGR 이미지로 변환하고, 픽셀 값이 0이 아닌 부분을 노란색(BGR: 0, 255, 255)으로 채움
        color_mask_bgr = np.zeros((*refined_mask.shape, 3), dtype=np.uint8)
        color_mask_bgr[refined_mask > 0] = [0, 255, 255]
        cv2.imwrite(mask_outp, color_mask_bgr)
    else: # 'white' 모드는 기존처럼 흑백(단일 채널) 마스크로 저장
        cv2.imwrite(mask_outp, refined_mask)
        
    print(f"✅ Processed: {os.path.basename(path)} ({mode}) -> Saved: {os.path.basename(mask_outp)}")

# 폴더 내 모든 이미지를 처리하는 함수
def process_images_in_dir(in_dir, output_base_dir, params, mode):
    print(f"📁 Processing images in: {in_dir}")
    print(f"📝 Saving results to: {os.path.join(output_base_dir, params['output_subdir'])}")
    
    search_path = os.path.join(in_dir, '*.*')
    imgs = sorted(glob.glob(search_path))
    
    if not imgs:
        print(f"🚨 Error: No image files found in '{in_dir}'. Check your path!")
        return
        
    for i, p in enumerate(imgs):
        process_image_file(p, output_base_dir, params, mode)
    print(f"✨ All images processed successfully for {mode} lines.")

# 이 스크립트가 직접 실행될 때 동작하는 메인 코드 블록
if __name__ == "__main__":
    # 명령줄 인자가 없으면 자동 실행 모드로 진입 (VS Code에서 바로 실행하는 경우)
    if len(sys.argv) == 1:
        print("--- 자동 실행 모드 (반사광 제거 개선) ---")
        
        # 1. 흰색 라인 마스크만 추출
        process_images_in_dir(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_BASE_DIR, WHITE_LINE_PARAMS, mode='white')
        
        # 2. 노란색 라인 마스크만 추출 (채도 하한 상향으로 반사광 제거)
        process_images_in_dir(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_BASE_DIR, YELLOW_LINE_PARAMS, mode='yellow')
        
        print("---------------------------------------")
    else:
        print("--- 명령줄 인자 실행 모드 (기본값 사용) ---")
        print("경고: 인자 없이 실행하려면 VSCode 버튼을 사용하거나 코드를 수정하세요.")