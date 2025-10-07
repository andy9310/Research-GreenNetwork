# Research - Dynamic clustering with hierachical multi-agent reinforcement learning in SDN-based network
## Situation and Task
1. 解決傳統啟發式演算法與單一強化學習模型在大規模網路中面臨的全域決策困境

## Environment
1. 200個節點2000個連結，40%的節點是邊緣節點，共80個host
2. 整體網路劃分為三個區域，並設定其流量高峰期分別發生在不同時段，以模擬真實網路中因地理位置與應用需求差異所產生的非同步高峰
3. 每個host每3-10s產生新的traffic flow ( 30-100 bytes ) (時間和大小隨機)，但分離尖峰時段的不同區間例如 尖峰時間每3-5s產生新的traffic flow ( 80-100 bytes )、 離峰時間每7-10s產生新的traffic flow ( 30-50 bytes )
4. 每個 traffic flow 具有隨機的 priority 等級 (1-6)、根據大部分ISP企業規範

## Architecture
1. 在一個大型拓樸中進行動態分群，分群的依據為(流量矩陣、拓樸形狀、服務級別占比)，各自群體內自行進行決策(開關連結)
2. 每個群體內部具有 Deterministic 決策來幫忙 (rule-based refinement)

## Evaluation 
與 ILP/MIP用求解器求解、Heuristic、強化學習DQN 等方法進行比較
比較方法與論文
1. ILP/MLP
2. Heuristic
3. 強化學習DQN

比較項目: 
1. 運算時間(延遲)(不包含訓練時間)
2. 節能效果與latency
   
## folder architecture
* train.py
* config.json
* env.py
* cluster.py
* algorithm.py
* compare (folder)
   * compare.py

## technique identify
### clustering model
   * k-means (topology、traffic、service)
### 

### deterministic algorithm
啟發式（heuristic）節能演算法，其操作方式如下：
每一個時間點，根據目前的網路狀態判斷是否需要調整網路結構。
在保證整個網路仍然保持連通（即每個節點仍能互相溝通）的前提下：
關閉 buffer 長度低於 [30%] 的節點周圍連結，且過去5秒內的buffer length 成長率小於[30%]，這些節點表示目前流量負載較低。
對於要關閉的連結中，保留那些 link usage（使用率）較高，大於[50%]的連結，其餘則關閉。
將原本這些關閉連結的流量轉導至仍然開啟的連結，但要確保導入後的連結使用率不超過預設門檻 [90%]，以防止流量過載（overload）。

