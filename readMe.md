# 强化学习从入门到放弃的资料 

##  目录

 - [书](#书)
  - [Reinforcement Learning: An Introduction](#reinforcement-learning-an-introduction) 
  - [Algorithms for Reinforcement Learning](#Algorithms-for-Reinforcement-Learning)
  - [OpenAI Spinning Up](#OpenAI-Spinning-Up)


 - [课程](#课程)

     - [基础课程](#基础课程)
       
       - [Rich Sutton 强化学习课程(Alberta)](#Rich-Sutton-强化学习课程（Alberta）)
       - [David Silver 强化学习课程（UCL）](#David-Silver-强化学习课程（UCL）)
       - [Stanford 强化学习课程](#Stanford-强化学习课程)
       - [UCL + STJU Multi-Agent Reinforcement Learning Tutorial](#Multi-Agent-Reinforcement-Learning-Tutorial)
     - [深度强化学习 (DRL) 课程](#深度强化学习-(DRL)-课程)
       
       - [台湾大学 李宏毅 （深度）强化学习](#台湾大学-李宏毅-（深度）强化学习)
       - [UCB 深度强化学习课程](#UCB-深度强化学习课程)
       - [CMU 深度强化学习课程](#CMU-深度强化学习课程)

## 书
### Reinforcement Learning: An Introduction
Richard Sutton and Andrew Barto, Reinforcement Learning: An Introduction 

> 第二版的最终版： [link](http://incompleteideas.net/book/the-book-2nd.html)，因为官方的是放在 Google Doc 上，所以我就下载了一个放在 Github 上，如有需要可以从 [这里](https://github.com/wwxFromTju/awesome-reinforcement-learning-zh) 自取。
>
> 已经可以准备买实体书了，和同学各自海淘了一本，还没有到手 -- 国外亚马逊， 国内的话，可以考虑京东和国内的亚马逊--不过会贵一些。

### Algorithms for Reinforcement Learning
 Csaba Szepesvari, Algorithms for Reinforcement Learning [link](http://www.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)

### OpenAI Spinning Up
[在线的文档](http://spinningup.openai.com/en/latest/)

[关于强化学习的基础介绍](http://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

[关于深度强化学习的建议](http://spinningup.openai.com/en/latest/spinningup/spinningup.html)

[代码部分](https://github.com/openai/spinningup/tree/master/spinup)

> 这个算是比较杂的书吧，有在线 doc + 对应的 code + 对应的练习（非常建议结合UCL的一起看，我大致过了一遍，蛮不错的。 * _但是没有提到下面的UCL，UCB的课，也没有提到上面 Sutton 的书，结合得看或许会更好_ *



## 课程
### 基础课程

#### Rich Sutton 强化学习课程（Alberta）
[课程主页](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLAIcourse/RLAIcourse2006.html)

>  这个比较老了，有一个比较新的在 Google 云盘上，我找个时间整理一下。

#### David Silver 强化学习课程（UCL）
> 这是 David Silver 大神 2015 在 UCL 开的课，现在感觉已经在 DeepMind 走向巅峰了，估计得等他那天想回学校培养学生才可能开出新的课吧。非常推荐入门学习，建立基础的RL概念。

[课程主页](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

对应课件：

Lecture 1: Introduction to Reinforcement Learning [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf)

Lecture 2: Markov Decision Processes [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf)

Lecture 3: Planning by Dynamic Programming [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)

Lecture 4: Model-Free Prediction [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf)

Lecture 5: Model-Free Control [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf)

Lecture 6: Value Function Approximation [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf)

Lecture 7: Policy Gradient Methods [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)

Lecture 8: Integrating Learning and Planning [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/dyna.pdf)

Lecture 9: Exploration and Exploitation [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf)

Lecture 10: Case Study: RL in Classic Games [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/games.pdf)

#### Stanford 强化学习课程

> 注：为 2018 spring 的课

[课程主页](http://web.stanford.edu/class/cs234/schedule.html)

对应课件：

Introduction to Reinforcement Learning [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l1.pdf)

How to act given know how the world works. Tabular setting. Markov processes. Policy search. Policy iteration. Value iteration [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l2.pdf)

Learning to evaluate a policy when don't know how the world works. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l3.pdf)

Model-free learning to make good decisions. Q-learning. SARSA. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l4.pdf)

Scaling up: value function approximation. Deep Q Learning. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l5.pdf)

Deep reinforcement learning continued. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l6.pdf)

Imitation Learning. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l7_annotated.pdf)

Policy search. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l8.pdf)

Policy search. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l9_updated.pdf)

Midterm review. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_midterm_review.pdf)

Fast reinforcement learning (Exploration/Exploitation) Part I. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l11.pdf)

Fast reinforcement learning (Exploration/Exploitation) Part II. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l12.pdf)

Batch Reinforcement Learning. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l13.pdf)

Monte Carlo Tree Search. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l14.pdf)

Human in the loop RL with a focus on transfer learing. [link](http://web.stanford.edu/class/cs234/slides/cs234_2018_l15.pdf)

#### Multi-Agent Reinforcement Learning Tutorial

> 注：因为在阿里广告这边实习，有幸和汪老师还有张老师做了篇论文。在过程中体会到汪老师的思维真的很活跃，很强。另外，张老师感觉是国内 CS 冉冉升起的新星，值得 follow 和关注！

 [课程主页](http://wnzhang.net/tutorials/marl2018/index.html)

 Fundamentals of Reinforcement Learning  [link](http://wnzhang.net/tutorials/marl2018/docs/lecture-1-rl.pdf)

 Fundamentals of Game Theory [link](http://wnzhang.net/tutorials/marl2018/docs/lecture-2a-game-theory.pdf)

 Learning in Repeated Games [link](http://wnzhang.net/tutorials/marl2018/docs/lecture-2b-repeated-games.pdf)

 Multi-Agent Reinforcement Learning [link](http://wnzhang.net/tutorials/marl2018/docs/lecture-3a-marl-1.pdf) [link](http://wnzhang.net/tutorials/marl2018/docs/lecture-3b-marl-2.pdf)



### 深度强化学习 (DRL) 课程

#### 台湾大学 李宏毅 （深度）强化学习
[课程主页](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)

视频可以在 B站 上看到：[link](https://www.bilibili.com/video/av24724071?from=search&seid=14814651069494196110)

#### UCB 深度强化学习课程

[课程主页](http://rll.berkeley.edu/deeprlcourse/)

> Update: 2018 fall（2018年秋季）

对应课件：

Introduction and Course Overview [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf)

Supervised Learning and Imitation [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf)

TensorFlow and Neural Nets Review Session (notebook) [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-3.pdf)

Reinforcement Learning Introduction [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf)

Policy Gradients Introduction [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)

Actor-Critic Introduction [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf)

Value Functions and Q-Learning [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf)

Advanced Q-Learning Algorithms [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf)

Advanced Policy Gradients [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf)

Optimal Control and Planning [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf)

Model-Based Reinforcement Learning [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf)

Advanced Model Learning and Images [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-12.pdf)

Learning Policies by Imitating Other Policies [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-13.pdf)

Probability and Variational Inference Primer [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf)

Connection between Inference and Control [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-15.pdf)

Inverse Reinforcement Learning [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf)

Exploration: Part 1 [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-17.pdf)

Exploration: Part 2 [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf)

Transfer Learning and Multi-Task Learning [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-19.pdf)

Meta-Learning [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf)

Parallelism and RL System Design [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-21.pdf)

Advanced Imitation Learning and Open Problems [link](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-22.pdf)

#### CMU 深度强化学习课程

> Update: fall 2018 (2018 秋季)

[2018 fall 的课程主页](http://www.andrew.cmu.edu/course/10-703/)

[2017的课程主页](https://katefvision.github.io/)

对应课件：

Introduction [link](https://www.cs.cmu.edu/~katef/DeepRLFall2018/lecture1_intro.pdf)

Markov decision processes (MDPs), POMDPs [link](https://www.cs.cmu.edu/~katef/DeepRLFall2018/lecture2_mdps.pdf)

Solving known MDPs: Dynamic Programming [link](http://www.andrew.cmu.edu/course/10-703/slides/lecture3_exactmethods-9-5-2018.pdf)

Policy iteration, Value iteration, Asynchronous DP [link](http://www.andrew.cmu.edu/course/10-703/slides/lecture4_valuePolicyDP-9-10-2018.pdf)

Monte Carlo Learning, Temporal difference learning, Q learning [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture5_MC_9-12-2018.pdf)

Temporal difference learning (Tom), Planning and learning: Dyna, Monte carlo tree search [link](http://www.andrew.cmu.edu/course/10-703/slides/TDshort-9-17-2018.pdf)

Deep NN Architectures for RL [link](https://www.cs.cmu.edu/~katef/DeepRLFall2018/lecture_NNarchitecturesforRL_katef.pdf)

Recitation on Monte Carlo Tree Search [link](https://www.cs.cmu.edu/~katef/DeepRLFall2018/MCTS_katef.pdf)

VF approximation, MC, TD with VF approximation, Control with VF approximation[link](https://www.cs.cmu.edu/~katef/DeepRLFall2018/lecture_FAkatef.pdf)

Deep Q Learning : Double Q learning, replay memory [link](https://www.cs.cmu.edu/~katef/DeepRLFall2018/lecture_DQL_katef2018.pdf)	

Policy Gradients [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_PG-10-1-2018.pdf) [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_PG-10-3-2018.pdf)

Advanced Policy Gradients [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_PG-NatGrad-10-8-2018.pdf)

Evolution Methods, Natural Gradients [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_async_evolution.pdf)

Natural Policy Gradients, TRPO, PPO, ACKTR [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_NaturalPolicyGradientsTRPOPPO.pdf)

Pathwise Derivatives, DDPG, multigoal RL, HER [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_DDPGMultigoalRL.pdf)

Exploration vs. Exploitation [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_Exploration-10-22-2018.pdf) [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_exploration.pdf)

Exploration and RL in Animals [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_exploration.pdf) [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_Brains_RL.pdf)

Model-based Reinforcement Learning [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_modelbasedRL.pdf)

Imitation Learning [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_Imitation_supervised-Nov-5-2018.pdf)

Maximum Entropy Inverse RL, Adversarial imitation learning [link](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_IRL_GAIL.pdf)

Recitation: Trajectory optimization - iterative LQR [link](https://katefvision.github.io/katefSlides/RECITATIONtrajectoryoptimization_katef.pdf)

Learning to learn, one shot learning[link](Learning to learn, one shot learning)

##  内容更新记录

2018-11-10：

1. 加入 OpenAI 的 spinningup
2. 加入台湾大学李宏毅的课
3. 加入 UCL 汪军老师 与 SJTU 张伟楠 老师 在 SJTU 做的 Multi-Agent Reinforcement Learning Tutorial  
4. update UCB 与 CMU 的 DRL 课到 2018 fall
5. update Sutton 的书到 final version
