# 困难集的构建

# 这个作为data的基本单位
class baseData:
    """data的基本单位，每一个data应该包含question，answer和多条交互存储链"""
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.interaction_chain = []


    


# 依照论文的说法，这种困难集应该是动态评估，动态构建的
class hardDataset:
    """以后就都叫hardDataset"""

    def build(self, dataset, model):


        return 
    
    def get_data(self, ):
        return 
    
    



# 因此，老规矩，先写测试，测试应该有这些内容

def test_difficult_dataset():

    # 初始化
    先能够评估困难集合，需要一个能够评估困难集合的接口
    因为我们的方法是RAG的方法，所以query和dataset与原来的QA是存在gap的
    ddataset = difficultDataset(
        dataset = dataset, 
        model = model, 
        tokenizer = tokenizer, 
    )

    for step in range(1000):
        假设这是在训练的迭代中

        那么应该可以取出困难集中的结果
        question, answer = ddataset.get_data()

        每隔eval_difficult_dataset步，进行困难集的重新评估（在原来的基础上）
        ddataset.update_self()



