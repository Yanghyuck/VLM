# 보안 체크리스트

## 즉시 조치 필요

### 1. DB 비밀번호 노출 — git 히스토리에 남음

`config.json` 을 `.gitignore` 로 옮겼지만 과거 커밋(`82862b7`)에 평문 비밀번호가 남아 있습니다.
GitHub 리포가 public 인 경우 **이미 노출**된 것으로 간주해야 합니다.

#### 권장 조치 (둘 중 하나)

**A. DB 비밀번호 변경 (가장 간단, 권장)**
```sql
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_strong_password';
FLUSH PRIVILEGES;
```
그리고 로컬 `config.json` 에만 새 비밀번호 반영.

**B. git 히스토리 재작성 (리포 이력 변조)**
```bash
# 1) git-filter-repo 설치
pip install git-filter-repo

# 2) config.json 을 전체 히스토리에서 제거
git filter-repo --path config.json --invert-paths --force

# 3) 강제 푸시 (※ 팀원이 있다면 사전 공지 필수)
git push origin --force --all
```
> 강제 푸시는 파괴적 작업입니다. 팀 공유 리포라면 A 방식을 권장합니다.

---

## 운영 환경 배포 전 확인

- [ ] `config.json` 의 `db.password` 를 환경변수로 주입하도록 변경 (`os.environ.get`)
- [ ] `api.allowed_origins` 를 실제 도메인으로 제한 (현재: localhost만)
- [ ] HTTPS 리버스 프록시 (nginx / Caddy) 앞단에 배치
- [ ] FastAPI 요청 로깅 + 비정상 접근 알림 설정
- [ ] rate limiting (slowapi) 추가
