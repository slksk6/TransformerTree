import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns




# 读取Excel表格
df = pd.read_excel('data/数据集.xlsx')

# 定义商品大类列表
product_categories = ['牛产品', '水产品', '鸡产品', '羊产品', '猪产品']

# 定义商品大类特征字典
category_features = {
    "牛产品": ['供应商ID', '厂号', '品种', '重量'],
    "水产品": ['等级', '申报类别', '切割方式'],
    "鸡产品": ['含肉率', '饲养方式', '原产地'],
    "羊产品": ['供应商ID', '等级', '申报类别'],
    "猪产品": ['供应商ID', '原产地']
}

# 创建一个字典来存储每个商品大类处理后的数据
processed_data = {}

# for category in product_categories:
#     # 筛选出当前商品大类的数据
#     df_category = df[df['商品大类'] == category]

#     # 复制数据，避免修改原始数据
#     df_processed = df_category.drop(columns=['商品大类']).copy()

#     df_processed['单价(原币)'] = df_processed['单价(原币)'] * df_processed['人民币汇率']


#     le = LabelEncoder()  # Create a new LabelEncoder for each category
#     for column in df_processed.columns:
#         if column not in ['采购时间', '单价(原币)']:
#             df_processed[column] = df_processed[column].astype(str)
#             df_processed[column] = le.fit_transform(df_processed[column])

#     # 将日期列转换为日期时间格式
#     df_processed['采购时间'] = pd.to_datetime(df_processed['采购时间'])

#     # 根据年月分组，计算其他列的平均值
#     #result_category = df_processed.groupby(df_processed['采购时间'].dt.to_period('M')).mean()  # Group by month
#     result_category = df_processed
#     # 添加商品大类列，所有值都为当前类别
#     result_category['商品大类'] = category

#     # 将结果存储到字典中
#     processed_data[category] = result_category


# # 将每个商品大类处理后的结果保存到单独的Excel表格中
# for category, result in processed_data.items():
#     result.to_excel(f'out_{category}.xlsx')



for category in product_categories:
    all_dfs = []
    for col in category_features.get(category, []): 
        torch.manual_seed(0)
        np.random.seed(0)

        # This concept is also called teacher forceing. 
        # The flag decides if the loss will be calculted over all 
        # or just the predicted values.
        calculate_loss_over_all_values = False

        # S is the source sequence length
        # T is the target sequence length
        # N is the batch size
        # E is the feature number

        #src = torch.rand((10, 32, 512)) # (S,N,E) 
        #tgt = torch.rand((20, 32, 512)) # (T,N,E)
        #out = transformer_model(src, tgt)
        #
        #print(out)

        input_window = 5
        output_window = 1
        batch_size = 100 # batch size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 对变量进行归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        class PositionalEncoding(nn.Module):

            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()       
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                #pe.requires_grad = False
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:x.size(0), :]
            

        class TransAm(nn.Module):
            def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
                super(TransAm, self).__init__()
                self.model_type = 'Transformer'
                
                self.src_mask = None
                self.pos_encoder = PositionalEncoding(feature_size)
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
                self.decoder = nn.Linear(feature_size,1)
                self.init_weights()

            def init_weights(self):
                initrange = 0.1    
                self.decoder.bias.data.zero_()
                self.decoder.weight.data.uniform_(-initrange, initrange)

            def forward(self,src):
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    device = src.device
                    mask = self._generate_square_subsequent_mask(len(src)).to(device)
                    self.src_mask = mask

                src = self.pos_encoder(src)
                output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
                output = self.decoder(output)
                return output

            def _generate_square_subsequent_mask(self, sz):
                mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                return mask



        # if window is 100 and prediction step is 1
        # in -> [0..99]
        # target -> [1..100]
        def create_inout_sequences(input_data, tw):
            inout_seq = []
            L = len(input_data)
            for i in range(L-tw):
                train_seq = np.append(input_data[i:i+tw][:-output_window], output_window * [0])
                train_label = input_data[i:i+tw]
                #train_label = input_data[i+output_window:i+tw+output_window]
                inout_seq.append((train_seq ,train_label))
            return torch.FloatTensor(inout_seq)

        def get_data():
            """
            time        = np.arange(0, 400, 0.1)
            amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))

            #series = pd.read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
            amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
            """
            # 读取Excel文件
            df = pd.read_excel(f'out_{category}.xlsx')


            # 获取时间序列和要预测的变量
            time = df['采购时间'].values
            amplitude = df[col].values

            amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)


            sampels = int(0.9 * len(amplitude))
            train_data = amplitude
            test_data = amplitude[sampels:]
            print(len(test_data))
            # convert our train data into a pytorch train tensor
            #train_tensor = torch.FloatTensor(train_data).view(-1)
            # todo: add comment..
            train_sequence = create_inout_sequences(train_data,input_window)
            train_sequence = train_sequence[:-output_window] #todo: fix hack?

            #test_data = torch.FloatTensor(test_data).view(-1)
            test_data = create_inout_sequences(test_data,input_window)
            test_data = test_data[:-output_window] #todo: fix hack?

            return train_sequence,test_data

        def get_batch(source, i,batch_size):
            seq_len = min(batch_size, len(source) - 1 - i)
            data = source[i:i+seq_len]    
            input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
            target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
            return input, target


        def train(train_data):
            model.train() # Turn on the train mode
            total_loss = 0.
            start_time = time.time()

            for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
                data, targets = get_batch(train_data, i,batch_size)
                optimizer.zero_grad()
                output = model(data)        

                if calculate_loss_over_all_values:
                    loss = criterion(output, targets)
                else:
                    loss = criterion(output[-output_window:], targets[-output_window:])
            
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                log_interval = int(len(train_data) / batch_size / 5)
                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                        'lr {:02.6f} | {:5.2f} ms | '
                        'loss {:5.5f} | ppl {:8.2f}'.format(
                            epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()

        def plot_and_loss(eval_model, data_source,epoch):
            eval_model.eval() 
            total_loss = 0.
            test_result = torch.Tensor(0)    
            truth = torch.Tensor(0)
            with torch.no_grad():
                for i in range(0, len(data_source) - 1):
                    data, target = get_batch(data_source, i,1)
                    # look like the model returns static values for the output window
                    output = eval_model(data)    
                    if calculate_loss_over_all_values:                                
                        total_loss += criterion(output, target).item()
                    else:
                        total_loss += criterion(output[-output_window:], target[-output_window:]).item()
                    
                    test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
                    truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
                    
            #test_result = test_result.cpu().numpy()
            len(test_result)

            pyplot.plot(test_result,color="red")
            pyplot.plot(truth[:500],color="blue")
            pyplot.plot(test_result-truth,color="green")
            pyplot.grid(True, which='both')
            pyplot.axhline(y=0, color='k')
            pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
            pyplot.close()
            
            return total_loss / i


        def predict_future(eval_model, data_source,steps):
            eval_model.eval() 
            total_loss = 0.
            test_result = torch.Tensor(0)    
            truth = torch.Tensor(0)
            _ , data = get_batch(data_source, 0,1)
            with torch.no_grad():
                for i in range(0, steps,1):
                    input = torch.clone(data[-input_window:])
                    input[-output_window:] = 0     
                    output = eval_model(data[-input_window:])                        
                    data = torch.cat((data, output[-1:]))
                    
            data = data.cpu().view(-1)
            

            pyplot.plot(data,color="red")       
            pyplot.plot(data[:input_window],color="blue")
            pyplot.grid(True, which='both')
            pyplot.axhline(y=0, color='k')
            pyplot.savefig('graph/transformer-future%d.png'%steps)
            pyplot.close()
                
        # entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
        # auch zu denen der predict_future
        def evaluate(eval_model, data_source):
            eval_model.eval() # Turn on the evaluation mode
            total_loss = 0.
            eval_batch_size = 1000
            with torch.no_grad():
                for i in range(0, len(data_source) - 1, eval_batch_size):
                    data, targets = get_batch(data_source, i,eval_batch_size)
                    output = eval_model(data)            
                    if calculate_loss_over_all_values:
                        total_loss += len(data[0])* criterion(output, targets).cpu().item()
                    else:                                
                        total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
            return total_loss / len(data_source)

        train_data, val_data = get_data()
        model = TransAm().to(device)

        criterion = nn.MSELoss()
        lr = 0.005 
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

        best_val_loss = float("inf")
        epochs = 1 # The number of epochs
        best_model = None

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(train_data)
            
            
            if(epoch % 10 == 0):
                val_loss = plot_and_loss(model, val_data,epoch)
                predict_future(model, val_data,200)
            else:
                val_loss = evaluate(model, val_data)
                
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_model = model

            scheduler.step() 

        start_date = '2022-07-01'
        end_date = '2022-12-31'

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        input_window = len(date_range)
        batch_size = 1
        feature_number = 1

        src = torch.randn(input_window, batch_size, feature_number)
        out = model(src)

        # 转换为NumPy数组
        output_np = out.detach().cpu().numpy()

        # 将数组重构为2D形状
        output_np_2d = output_np[:, 0, :]
        output_np_2d = scaler.inverse_transform(output_np_2d)


        # 转换为DataFrame对象
        _df = pd.DataFrame(output_np_2d, columns=[col])

        # 将当前商品大类的 DataFrame 添加到列表中
        all_dfs.append(_df)

    # 将所有 DataFrame 合并为一个
    final_df = pd.concat(all_dfs, ignore_index=True)

    # 将合并后的 DataFrame 写入同一个 Excel 文件
    final_df.to_excel(f'output_{category}.xlsx', index=False, engine='openpyxl')



    # df = pd.read_excel("out.xlsx")

    # df_1 = df
    # df_train = df_1
    # #df_train = df_1.iloc[:int(len(df_1) * 0.9)]
    # X_train = df_train[['等级', '到达港', '供应商ID']].values
    # Y_train = df_train['单价(原币)']

    # # 读取测试集数据
    # df_test = df_1.iloc[int(len(df_1) * 0.9):]
    # X_test = df_test[['等级', '到达港', '供应商ID']].values
    # Y_test = df_test['单价(原币)']


    # # 决策树建模预测
    # regressor = DecisionTreeRegressor(random_state=0).fit(X_train, Y_train)
    # y_pred = regressor.predict(X_test)

    # # 可视化部分
    # sns.set(font_scale=1.2)
    # plt.rcParams['font.sans-serif'] = 'SimHei'
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rc('font', size=14)
    # plt.plot(list(range(0, len(X_test))), Y_test, marker='o')
    # plt.plot(list(range(0, len(X_test))), y_pred, marker='*')
    # plt.legend(['真实值', '预测值'])
    # plt.title('冷链库决策回归树预测值与真实值的对比')
    # plt.show()

    # x_pred = pd.read_excel('data/features.xlsx')
    # x_pred = x_pred[['等级', '到达港', '供应商ID']].values
    # y_pred1 = regressor.predict(x_pred)


    # data = {
    #     '单价': y_pred1,
    # }
    # df_output = pd.DataFrame(data)
    # # 将DataFrame保存到Excel文件中的某一列，例如第一列
    # df_output.to_excel('data/result.xlsx', startcol=0, index=False, engine='openpyxl')


