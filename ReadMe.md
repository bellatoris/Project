# 졸업 프로젝트

## 3월 4일
목표를 정하다: 이번 졸업 프로젝트의 목표로 6-DoF camera pose를 5개 정도의 연속된 사진을 넣어서 마지막 사진 혹은 첫번째 사진을 제외한 4개의 사진의 6-DoF camera pose를 구하는 CNN + RNN 구조의 Neural Network를 만들고자 한다. 

목표의 이유: Self Localization은 컴퓨터 비전에서 매우 중요한 문제이며, 움직이는 로봇이나 네비게이션, 증강 현실등에서 사용된다. 이것을 Learning problem으로써 해결한다는 의의가 존재한다.

PoseNet으로 부터 많은 intuition을 얻었고, DeMoN은 learning schedule이나 네트워크로 normal, depth, optical flow, 6-DoF camera pose를 모두 구하기 때문에 너무 복잡하다고 생각되어 (또한 쓸모없는 것들이 너무 들어간다 생각되어), 6-DoF camera pose만을 구하는 문제로 접근하기로 하였다.

### PoseNet
PoseNet은 하나의  Scene 혹은 Landmark를 정하여, 그 Landmark를 찍은 비디오로부터 뽑은 프레임과 각각 프레임에 해당하는 6-DoF camera pose (고전적인 컴퓨터 비전 방법으로 구하였다)을 target으로 하는 training set으로 만들어 Convolution Neural Network를 training 시킨다. 그 후 그 Landmark를 찍은 임의의 사진을 CNN에 input으로 넣으면 output으로 그 사진을 찍은 camera의 6-DoF camera pose를 내놓는다. 이 때 CNN은 ImageNet dataset을 사용하여 classification 문제를 해결하도록 pretrained 된 GoogleNet을 사용하였다.

**Transfer Learning:** GoogleNet은 분명 classification 문제를 해결하도록 pretrained 되어 있었지만, softmax classifier를 제거하고, (1024, 2048), (2048, 7) 두 개의 fully connected layer를 붙임으로써 pose를 regression하는 훌륭한  regressor가 되었다. 즉 classification이 pose를 regression하는 문제를 풀기위한 시작점으로 적절한지는 obvious하지 않으나, pose에 상관없이 classification하기 위해 classifier network가 pose를 tracking 해야 한다는 possible explanation이 있다고 저자들은 설명 한다. 나 또한 이것이 일리가 있다고 보며, 실제로 꽤나 좋은 성능을 보여주었다. 최근 실험 결과는 마지막 fully connected 전에 LSTM을 사용함으로써 state-of-art의 결과에 꽤나 접근했다.

**그러나**, PoseNet은 한 Network가  한 Scene (Landmark) 밖에 배우지 못하며, relocalization 문제 밖에 풀지 못해, 실제 SLAM을 대체 할수 없다. 저자들은 finite neural network가 learning을 통해 localize할 수 있는 physical area는 한계가 있는게 분명 하다고 주장한다.

그렇다면 physical area를 배우는게 아니라 여러장의 프레임이 들어왔을 때 첫번째 사진에 대한 camera pose를 구하게 하는 문제로 바꿔보면 어떨까? 이 문제가 충분히 가능하다고 생각되는 이유는 다음과 같다.

![PoseNet](./Picture/PoseNet.png)

위의 그림을 보면 알 수 있듯이, Place dataset (Scene classification dataset)만을 learning시킨 GoogleNet의 feature vector도 (마지막 softmax classifier 들어가긴 전 feature vector라고 예상됨) 시간대 별로 clustering되어있다는 점에서 CNN자체가 pose정보를 가지고 있다는 것을 알 수 있다. 또한 다른 지역을 learning 시킨 PoseNet도 (1024, 2048) fully connected layer를 통과한 feature vector가  매우 잘 정렬 되어있다는 점에서 마지막 pose prediction layer가 해당 scene에 overfitting되어 있어서 그렇지 그 전 feature는
frame별 pose의 정보를 매우 잘 표현할 수 있다고 본다. 즉 마지막 prediction layer를 scene에 overfitting 시키지 않고, 첫 번째 사진에 대해서만 pose를 뽑게 한다면 충분히 generalization 될 수 있지 않을까? 물론 해봐야 알겠지만...

### RNN
이런 6-DoF camera pose도 카메라가 시간에 따라 움직인 pose를 구한다는 점에서 spatio-temporal problem이다. 이런  spatio-temporal problem의 예로 video captioning이나 video action classification등이 있는데, 이런 문제들은 3D-Convolution이나, RNN을 같이 사용함으로써 많은 성과를 이뤄냈다. 여기서 사용한 방법들을 사용하면, 내가 풀고자 하는 문제도 도움을 얻을 수 있을 것이라 생각하였다. 

* 3D-Convolution (이하 C3D)는 transfer learning을 하기에는 적합하지 않다고 생각하여 (사실 제대로 생각 안해봄) 제외 하였지만, 지금 C3D가 video captioning에서 state-of-art에 있으므로 C3D의 사용도 생각해 보아야 한다. 논문도 아직 안읽어봄, 논문을 어서 읽자.
* GRU-RCN, convolution filter를 GRU의 weight들로 사용하자는 획기적인 방법이다. 그러나 너무 많은 parameter가 존재하여, 적은 training set으로 learning시키기에는 무리가 있어보이며, VGG-RNN에 비해 그렇게 차이나는 성능을 내지는 못한다.
* VGG-RNN, ConvNet으로 feature를 뽑고 마지막 feature을 RNN에 feeding한다. 이는 Video Capitioning Problem에서 매우 좋은 성능을 냈으며, (관련 논문 읽어 볼것!) slow fusion이나, late fusion같은 3D-Convolution들 보다도 월등한 성능을 보여줬다. 또한 구현도 그리 어렵지 않아 보인다. 이 구조를 base-line으로 삼아 우선 빨리 구현해 본다음에 성능을 보자. 물론 이 구조도 hyper-parameter들이 많이 존재하고(LSTM, GRU, pure RNN, 혹은 many-to-many, many-to-one등) 아직 RNN을 잘 이해하지 못하였지만 **창현씨** 말대로 빨리 구현해서 성능을 보는게 중요하다고 생각된다. 

### 우선순위
1. Res-Pose-Net 구현해서 성능 한번 볼것! dataset 만드는 연습도 될듯
2. Res-RNN을 빨리 구현해서 성능을 빨리 볼것 
3. 매일 매일 관련논문 한편씩 읽을것!!
4. 생각을 잘 정리해 둘것!