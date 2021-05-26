# Japan_GDP_data_analysis
일본 GDP 예측과 데이터 분석 파일 (2020년 고등학생 때 한 자료여서 지금도 많이 부족한데 더욱더 부족하다..)

# 개요
동아리에서 만든 자판기의 보급대수가 GDP 추세에 따르기 때문에 GDP가 어떻게 나아갈지 알면 자판기 사업 전망도 보인다. 이러한 취지로 시작한 프로그래밍이다.
사이킷런 선형회귀(linear regression)을 사용해서 간단하게 표에 선을 그어서 어떻게 나아갈지만 보여준다.

# 그래프
![gdp](https://user-images.githubusercontent.com/80466735/119701467-51fbf680-be8f-11eb-83ab-1468526d4c55.jpg)<br>
사진1. GDP 

![gdp성장률](https://user-images.githubusercontent.com/80466735/119701468-532d2380-be8f-11eb-85f2-6bee53456378.jpg)<br>
사진2. GDP 성장률
![소비자물가지수](https://user-images.githubusercontent.com/80466735/119701470-532d2380-be8f-11eb-94c7-41882411d0cc.jpg)<br>
사진3. 소비자 물가지수
![수익 선형회귀](https://user-images.githubusercontent.com/80466735/119701471-53c5ba00-be8f-11eb-9120-ac618d0d8685.jpg)<br>
사진4. 수익
![수출](https://user-images.githubusercontent.com/80466735/119701472-545e5080-be8f-11eb-9d14-11513c0e67b7.jpg)<br>
사진5. 수출
![자판기 보급대수](https://user-images.githubusercontent.com/80466735/119701473-545e5080-be8f-11eb-8e11-a9f036ca5d87.jpg)<br>
사진6. 자판기 보급대수
<br>
GDP성장률을 제외하고 모두 상승하는 모습이다. 여기까지가 고등학교 시절에 했던 자료이다. 추가적으로 자판기 보급대수와 비교하며 자판기 시장의 전망이 좋다는 말을 했었다.  <br>

# 기울기(coef)
코드에 line_fitter 변수에 LinearRegression(선형회귀)가 들어가있다. 즉 분석 결과(직선)가 들어가 있어서 거기에 .coef를 붙여서 linear_fitter.coef만 해도 간단히 그래프의 기울기를 알수가있다. 기울기를 통해서 그 변수의 성장을 얼마나 급격하게 할지 아니면 내려갈지를 순위를 매기는데 객관적인 지표 제공이 가능해진다. <br>
![image](https://user-images.githubusercontent.com/80466735/119702757-c5523800-be90-11eb-8b73-596e0f7bd46a.png)
![image](https://user-images.githubusercontent.com/80466735/119703086-2548de80-be91-11eb-920b-e79ebda53098.png)
![image](https://user-images.githubusercontent.com/80466735/119703141-3265cd80-be91-11eb-80b8-fc237bf98122.png)
![image](https://user-images.githubusercontent.com/80466735/119703170-3b569f00-be91-11eb-9781-7ae5ff2ccd1c.png)
![image](https://user-images.githubusercontent.com/80466735/119703186-41e51680-be91-11eb-9ca3-a2827c7f7551.png)<br>
하지만 이것은 각 데이터 간의 범위가 같을때 비교가 가능하다. 이 기울기 같은 경우 자료들마다 범위가 다르기 때문에 의미가 없다. 그래서 여기에 Min-Max 혹은 Z-score를 추가해서 데이터 정규화를 해야한다.<br>
(X - MIN) / (MAX-MIN) = 0~1사이의 값이 된다. 예를 들어 최대값 100 최솟값 10인 자료에서 X는 40을 정규화한다면 (40-10)/(100-10)=30/90=1/3으로 0.333...이 된다.<br>
![image](https://user-images.githubusercontent.com/80466735/119703940-0ac33500-be92-11eb-8a0d-6c7e9e4be80b.png)
![image](https://user-images.githubusercontent.com/80466735/119704268-67265480-be92-11eb-982a-02d39985d538.png)
![image](https://user-images.githubusercontent.com/80466735/119704286-6ee5f900-be92-11eb-82a0-3477bd63601b.png)
![image](https://user-images.githubusercontent.com/80466735/119704305-760d0700-be92-11eb-84dc-af29c2093d65.png)
![image](https://user-images.githubusercontent.com/80466735/119704326-7d341500-be92-11eb-8efb-1067c263440b.png)
이렇게 데이터들이 0~1사이로 표현이 되기 때문에 기울기도 그에 영향을 받아서 한눈에 보기가 더 쉬워진다. <br>
참고-http://hleecaster.com/ml-normalization-concept/
<br>
# 팩트풀니스, 이 자료의 문제점
 고등학교때 진행한 이 결과들의 문제점을 지적하고자한다.<br>
 1. 팩트풀니스(직선본능)<br>
  고2때 팩트풀니스 책을 읽었었다. 거기서 나온 내용중 직선본능을 지적한다. 인생이나 여러가지 원인들은 1차함수 그래프가 아니라 2차, 3차 등처럼 계속 변환이 있다는 내용이다. 그렇듯 저 그래프들 전부 1차함수 직선으로 나와있다. 우리가 이런 데이터를 보고선 투자를 한다거나 의사결정을 하기에는 너무 쓰레기 데이터다. 그냥 연습용 데이터에 불가하다는 것을 의미한다.<br>
 2. 단순한 흐름<br>
  GDP라는 경제지표는 단순히 흐름으로 예측하기에는 무리가 있다. 다양한 변수들이 GDP(어떻게 접근할지도 정해야한다. 그다음 변수를 고민해야한다.)에 어떤 영향을 주는지 가중치 알기위해 세세히  알아봐야한다. 사실 파고들다보면 끝도 없어서 무리가 있다. 하지만 이 데이터는 그냥 GDP 흐름만 보고 예측한거기 때문에 좀 별 의미 없다. 
  
# 결론
 초심자가 하는 데이터 분석이기 때문에 그냥 연습용으로 봐주면 좋을 거 같다.
