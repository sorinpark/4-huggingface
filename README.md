# 4-huggingface
4기 허깅페이스 세션입니다.

## 제출 방법

1. 자신의 깃허브 아이디로 브랜치 만든 다음에 개인 폴더 만들어주세요.
2. 작업 하시고 커밋 후 오리진으로 푸시해주시면 됩니다.
3. main 으로 풀리퀘 날려주시면 됩니다. merge 는 하시면 안돼요!

### 예시

```bash
!git clone https:/[토큰값]@github.com/HateSlop/4-huggingface.git  # 클론
!cd 4-huggingface # 프로젝트 루트로 이동
!git checkout -b '본인 브랜치명' # 브랜치 생성 (본인의 브랜치, 폴더 등 생성)
!mkdir '본인 폴더명' # 개인 폴더 만들기
!cd '본인 폴더명' # 개인 폴더로 이동
# 작업을 진행해주세요
!git add . # 작업 후 add
!git commit -m "[feat] ~~" # 커밋
!git push origin '본인 브랜치명' # 오리진에 푸시
```

### 폴더구조

```bash
.
├── README.md    # 프로젝트 설명 리드미
├── huggingface_basics.ipynb
├── review_신라스테이 해운대.json
├── transformer.ipynb # 과제
└── 본인_폴더명/                # 개인 폴더
    └── (개인 작업물) 
```

## 커밋 컨벤션

feat: 새로운 기능 추가  
fix: 버그 수정  
docs: 문서 수정  
style: 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우  
refactor: 코드 리팩토링  
test: 테스트 코드, 리팩토링 테스트 코드 추가  
chore: 빌드 업무 수정, 패키지 매니저 수정
