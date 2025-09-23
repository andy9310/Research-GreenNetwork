# Research - Dynamic clustering with hierachical multi-agent reinforcement learning in SDN-based network
## Situation and Task
1. 解決傳統啟發式演算法與單一強化學習模型在大規模網路中面臨的全域決策困境
## Environment
1. 200個節點2000個連結，40%的節點是邊緣節點，共80個host
2. 整體網路劃分為三個區域，並設定其流量高峰期分別發生在不同時段，以模擬真實網路中因地理位置與應用需求差異所產生的非同步高峰
3. 每個host每3~10s產生新的traffic flow ( 30~100 bytes ) (時間和大小隨機)，但分離尖峰時段的不同區間例如 尖峰時間每3~5s產生新的traffic flow ( 80~100 bytes )、 離峰時間每7~10s產生新的traffic flow ( 30~50 bytes )
4. 每個 traffic flow 具有隨機的 priority 等級 (1~6)、根據大部分ISP企業規範

## Architecture
1. 在一個大型拓樸中進行動態分群，分群的依據為(流量矩陣、拓樸形狀、服務級別占比)，各自群體內自行進行決策(開關連結)
2. 每個群體內部具有Deterministic決策來幫忙 (rule-based refinement)

## Evaluation 
與 ILP/MIP用求解器求解、Heuristic、強化學習DQN 等方法進行比較
比較方法與論文
1. ILP/MLP
2. Heuristic
3. 強化學習DQN

比較項目: 
1. 運算時間(延遲)(不包含訓練時間)
2. 節能效果與latency
   
## settings 



