import os, sys
import cnn_model
import paddle
try:
    paddle.enable_static()
except:
    print('\n正在使用低版本的飞桨框架！')
import reader as reader
import paddle.fluid as fluid


test_list = "test.list"
train_list = "train.list"
save_path = "model_infer"


test_list = '../data/' + test_list
train_list = '../data/' + train_list
save_path = '../model/' + save_path

crop_size = 120
resize_size = 120

image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='float32')

model = cnn_model.cnn_model(image)

cost = fluid.layers.square_error_cost(input=model, label=label)
avg_cost = fluid.layers.mean(cost)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)

opts = optimizer.minimize(avg_cost)

# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader(train_list, crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader(test_list, crop_size), batch_size=32)

# 定义执行器

# 使用GPU训练
place = fluid.CUDAPlace(0)
print('请注意，正在使用GPU训练！')

# 使用CPU
# place = fluid.CPUPlace()
# print('请注意，正在使用CPU训练！')
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练
all_test_cost = []
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f' %
                  (pass_id, batch_id, train_cost[0]))

    # 进行测试
    test_costs = []

    for batch_id, data in enumerate(test_reader()):
        test_cost = exe.run(program=test_program,
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    all_test_cost.append(test_cost)

    # test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost))
    # save_path = 'infer_model/'
    # 保存预测模型

    if min(all_test_cost) >= test_cost:
        fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], main_program=test_program,
                                      target_vars=[model], executor=exe)
        print('finally test_cost: {}'.format(test_cost))
