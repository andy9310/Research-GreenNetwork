network simulation 需要修改: 
1.        self.target_switches = target_switches
2.        self.num_actions = 2**11
3.        def _create_action_map(self): 更換為對應的 bit 數量
                actions.append(np.array([int(x) for x in f"{i:011b}"]))  # Convert index to binary

train_dqn_ 需要修改:
1.          elif method == "greedy_on":
                 action = 2047
2.          target_switches_ = [4, 9, 10, 15]


reward function 可以把屬於 constraint 的設定很大


Note: 學出來的已經不是 optimal
下一步需要再多一個 Node...(看到底多難學)