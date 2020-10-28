carla2gym manual

https://github.com/praveen-palanisamy/macad-gym. 을 참고해서 코드 작업하였으나, 개발하면서 완전 그냥 다른 코드라고 보면 되므로, 하다 막히는게 있으면 전화주시면 언제든 환영^^.

[python 실행 arguments]

--controller-coef : selector(구 controller)의 kl divergence threshold

--expert-reward-weight : expert demo 로부터 얻은 discr reward의 weight

--extr-reward-weight : environment reward의 weight

(commands.txt안에 실행 명령어 있음, 추가적인 인자에 대해서는 형들이 더 잘 아실거 같으므로 스킾)

[carla2gym 사용하기]
- 가상환경(python 3.5) 세팅, git clone (exploration branch)
- carla 0.9.6 설치
- pytorch 1.3.1, tensorflow-gpu 1.14.0 설치
- baselines 설치, custom-tasks 설치 ($ pip install -e .)
- main.py에 import custom_tasks 없으면 추가
- ~/.bashrc에 PYTHONPATH 추가

위까지가 기본.

- custom-tasks/custom_tasks/envs/ 안에 carla env있음.
- 그 중 가장 default가 CarlaUrbanIntersection8425.py (이하 8425.py)이고 envs = gym.make("CarlaUrbanIntersection8425-v0") 로 호출
- 8425.py 내부 함수 설명

def reset : 차량 위치를 reset.

def step(action) : 주어진 action 처리 후 next_obs, reward, done, info return.

def pause, def resume : carla 일시 정지 후 resume for synchronizing.

