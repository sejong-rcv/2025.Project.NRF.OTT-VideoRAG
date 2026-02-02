# Long Video Understanding Analysis based on LLaVA-NeXT

본 저장소는 신진연구 과제  
**「사용자 맞춤형 콘텐츠 미리보기 및 요약 자동 생성 연구를 위한 대형모델 활용 기술 개발」**  
의 일환으로 수행된 연구 결과를 정리한 것이다.

본 연구의 목적은 **장기간(Long Video) 영상 이해 환경에서 대형 멀티모달 모델의 성능 특성**을 분석하고,  
프레임 수 증가가 정확도 및 예측 불확실성(Entropy)에 미치는 영향을 정량적으로 이해하는 데 있다.

---

## Overview

- **Base Framework**
  - LLaVA-NeXT (Large Language and Vision Assistant – Next Generation)
- **Baseline Method**
  - Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension
- **Dataset**
  - Video-MME (Short / Medium / Long)
- **Metrics**
  - Accuracy
  - Entropy (prediction uncertainty)

본 저장소는 **모델 개발 또는 재배포 목적이 아닌**,  
LLaVA-NeXT 기반 장시간 영상 이해 성능 분석을 위한 **실험·분석 코드 및 결과 정리**를 목표로 한다.

---

## Key Findings

- 프레임 수(logit)가 증가한다고 해서 **항상 성능이 향상되지는 않음**
- Medium / Long 비디오 구간에서는  
  특정 프레임 수(예: 24, 28)에서 최적 성능이 관찰됨
- 과도한 프레임 입력은
  - 정보 중복
  - 노이즈 증가
  - 모델 추론 부담
  으로 인해 성능 저하로 이어질 수 있음
- Entropy 분석 결과 또한 프레임 수 증가가
  **항상 예측 불확실성 감소로 이어지지 않음**을 확인함

이는 **장시간 영상 이해를 위한 프레임 선택 및 요약 전략의 중요성**을 시사한다.

---

## Upstream Projects

본 연구는 아래 오픈소스 프로젝트 및 연구 결과를 기반으로 진행되었습니다.

- **LLaVA-NeXT**
  - GitHub: https://github.com/LLaVA-VL/LLaVA-NeXT
  - Blog: https://llava-vl.github.io/blog/
- **Video-RAG**
  - Paper: https://arxiv.org/abs/2411.13093

---

## Citation

본 저장소는 아래의 연구에 도움을 받았습니다.

```bibtex
@misc{luo2024videoragvisuallyalignedretrievalaugmentedlong,
      title={Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension}, 
      author={Yongdong Luo and Xiawu Zheng and Xiao Yang and Guilin Li and Haojia Lin and Jinfa Huang and Jiayi Ji and Fei Chao and Jiebo Luo and Rongrong Ji},
      year={2024},
      eprint={2411.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13093}, 
}
