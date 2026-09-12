# 실험 진행 상황

전체 연구 설계는 [`experiment_context.md`](./experiment_context.md) 참고. 여기서는 진행 단계만 추적한다.

Deliverable Order (설계 문서 7절 기준) — 각 단계는 이전 단계에서 kriging이 해당 케이스에서 정상적으로 calibrate됨을 확인한 뒤에만 다음으로 넘어간다. 특정 케이스에서 kriging 자체가 calibrate되지 않으면 그건 결과가 아니라 케이스 설정 오류로 간주.

- [ ] 1. Base case 재현 — 기존 single-model 결과를 새 코드베이스에서 재현
- [ ] 2. Variogram range 변화 실험
- [ ] 3. Nugget 변화 실험
- [ ] 4. Extrapolation / data configuration 변화 실험
- [ ] 5. Anisotropy 변화 실험

## 로그
| 날짜 | 단계 | 상태 | 비고 |
|---|---|---|---|
| 2026-09-11 | - | 연구 handoff 문서 수령, 프로젝트에 반영 | 아직 구현 시작 전 |
