import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from transformers import AutoModelForCausalLM, AutoTokenizer
#from fastapi import FastAPI
import torch
import torch.nn as nn
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from model import load_data, Standardization, set_random_seed
from utils import *
from websocket_transfer import RealTimeTransfer
from rag import retrieve, rerank
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import queue
import threading
import copy

# 定义全局发送队列
send_queue = queue.Queue()
# 添加大模型处理队列
model_queue = queue.Queue()

def model_worker():
    """独立线程：专门处理大模型请求"""
    while True:
        try:
            # 从队列获取请求
            prompt, transfer, model_dict, results_history = model_queue.get(timeout=0.1)
            
            # 处理大模型请求（这会阻塞，但只在独立线程中）
            text = generate_text(
                model=model_dict["llm_model"],
                tokenizer=model_dict["tokenizer"],
                prompt=prompt,
                rag=True,
                embedding_model=model_dict["embedding_model"],
                chromadb_collection=model_dict["chromadb_collection"],
                cross_encoder=model_dict["cross_encoder"],
                device=model_dict["device"]
            )
            results_history["response"].append(text)
            print(f"大模型输出:\n{text}")
            
            # 发送大模型响应
            send_message_async(text)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Model worker error: {e}")

def sender_thread(transfer: RealTimeTransfer, send_queue: queue.Queue):
    """独立线程：负责发送 websocket 消息"""
    while True:
        try:
            msg_type, msg_data = send_queue.get(timeout=0.1)
            if msg_type == "text":
                transfer.frame = msg_data
            elif msg_type == "list":
                transfer.list_data = msg_data
            print("已发送:", msg_data)
            transfer.send = True
            # 非阻塞等待，避免卡死
            while transfer.send:
                time.sleep(0.01)
        except queue.Empty:
            continue


def send_message_async(message: str):
    """放入队列，异步发送 text 消息"""
    send_queue.put(("text", message))


def send_soh_async(message: list):
    """放入队列，异步发送 list 数据"""
    send_queue.put(("list", message))

def data_process(data_path):
    """
    对数据进行加噪声干扰,得到包含异常数据和正常数据的合集
    :param data_path: 处理后的json路径
    :return: 处理后的prompts
    """
    # Load Dataset
    data_list = load_data(os.path.join(os.path.dirname(data_path), '无异常.txt'), add_noise=False)
    dlen = len(data_list)
    # 加载异常数据
    #bad_data = load_data(data_path, add_noise=False)  # 直接读取异常数据
    process = AnomalyProcessor(data_list)
    bad_data = process.noise_adder(data_list, noise_level=0.25, noise_rate=1.0)
    nlen = 256
    nlen2 = 0
    for i in range(dlen):
        if i >= nlen and i < 2 * nlen:
            data_list[i] = bad_data[i - nlen]
            nlen2 += 1
    print(f'Successfully Adding {len(bad_data)} Data to Normal Data!') 
    return data_list, nlen, nlen2

def realtime_soh_plot(results: dict, max_len: int):
    '''
    绘制实时SOH预测值图
    :param soh_list: 实时SOH预测值列表
    :param max_len: 绘图最大长度
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["soh"], label="SOH", color='b')
    ax.plot([thres * 100 for thres in results["threshold"]], color='black', alpha=0.5, label='Threshold')
    # 各类预警输出
    # 故障预警
    anomaly_flags = [True if warning == 1 else False for warning in results["warning"]]
    anomaly_indices = [i for i, flag in enumerate(anomaly_flags) if flag]
    if anomaly_indices:
        ax.scatter(
                anomaly_indices, 
                [results["soh"][i] for i in anomaly_indices],
                color='red', marker='x', label='Detected Anomalies'
            )
    # 下滑预警
    decrease_flags = [True if warning == 2 else False for warning in results["warning"]]
    decrease_indices = [i for i, flag in enumerate(decrease_flags) if flag]
    if decrease_indices:
        plt.scatter(
            decrease_indices,
            [results["soh"][i] for i in decrease_indices],
            color='orange', marker='^', label='Decrease Warning'
        )
    # 长期临界预警
    critical_flags = [True if warning == 3 else False for warning in results["warning"]]
    critical_indices = [i for i, flag in enumerate(critical_flags) if flag]
    if critical_indices:
        plt.scatter(
            critical_indices,
            [results["soh"][i] for i in critical_indices],
            color='green', marker='*', label='Long-Term Critical Warning'
        )
    # 添加SOH与RUL注释
    if results['soh'] and results['rul']:
        annotation_text = f"SOH = {results['soh'][-1]:.2f}\nRUL = {results['rul'][-1]}"
        ax.text(
            0.99, 0.01, annotation_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )
    ax.set_xlim(0, max_len)
    ax.set_ylim(0, 102)
    ax.set_title("Real-Time SOH Prediction")
    ax.set_xlabel("Steps")
    ax.set_ylabel("SOH Value")
    ax.grid(True)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('./outputs/soh_prediction.png')
    plt.close()

class RealTimeSOHPredicter:
    def __init__(self, model_dict, num_detect_samples, feature_columns):
        self.device = model_dict["device"]
        self.llm_model = model_dict["llm_model"]
        self.tokenizer = model_dict["tokenizer"]
        self.soh_detector = model_dict["soh_detector"]
        self.embedding_model = model_dict["embedding_model"]
        self.cross_encoder = model_dict["cross_encoder"]
        self.chromadb_collection = model_dict["chromadb_collection"]
        self.feature_columns = feature_columns
        self.num_detect_samples = num_detect_samples
        self.remembered_max_result = 128
        self.history_window = []
        self.results = {
            "soh": [],
            "threshold": [],
            "warning": [],
            "rul": [],
            "rul_info": [],
        }

    def add_sample(self, feature):
        self.history_window.append(feature)
        if len(self.history_window) > self.num_detect_samples:
            self.history_window.pop(0)
        
        if len(self.history_window) < self.num_detect_samples:
            return None
        
        # 构造滑动窗口
        batch_features = np.vstack(self.history_window)
        soh, thres, warning, rul, rul_log, warning_log = self.soh_detector.detect_soh(batch_features)
        self.results["soh"].append(soh)
        self.results["threshold"].append(thres)
        self.results["warning"].append(warning)
        self.results["rul"].append(rul)
        self.results["rul_info"].append(rul_log)
        if len(self.results["soh"]) > self.remembered_max_result:
            self.results["soh"].pop(0)
            self.results["threshold"].pop(0)
            self.results["warning"].pop(0)
            self.results["rul"].pop(0)
            self.results["rul_info"].pop(0)

        #print(self.soh_detector.log_info[-1])
        
        return {
            'soh': soh,
            'threshold': thres,
            'warning': warning,
            'rul': rul,
            'log': rul_log + "\n" + warning_log
        }
        
class RealTimeDataReader:
    def __init__(self, file_path, feature_names=None, add_noise=False):
        self.feature_names = feature_names or [
            'Motor1', 'Motor2', 'Motor3', 'Motor4', 'Motor5', 'Motor6', 'Motor7', 'Motor8',
            'Motor9', 'Motor10', 'Motor11', 'Motor12', 'Motor13', 'Motor14', 'Motor15', 'Motor16',
            'Motor17', 'Motor18', 'Motor19', 'Motor20', 'Motor21', 'Motor22', 'Motor23', 'Motor24',
            'Accelx', 'Accely', 'Accelz', 'AngAcx', 'AngAcy', 'AngAcz', 'Eulerx', 'Eulery', 'Eulerz',
            'Voltage', 'Current', 'Power', 'Battery',
            'GPS_longitude', 'GPS_latitude', 'GPS_altitude'
        ]
        self.index = 0
        #self.data_list = load_data(file_path, add_noise=add_noise)
        self.data_list, self.nlen, self.nlen2 = data_process(file_path)
        self.total_index = len(self.data_list)
        
    def read_next(self):
        """读取下一行样本"""
        if self.index < self.total_index:
            row = self.data_list[self.index]
            self.index += 1
            return row
        else:
            return None
        
def generate_text(model, tokenizer, prompt: str, device="cpu", rag=False, embedding_model=None, cross_encoder=None, chromadb_collection=None):
    '''调用本地大模型生成文本'''
    start_time = time.time()
    if rag:
        top_k = 7
        retrieved_chunks = retrieve(prompt, top_k, chromadb_collection, embedding_model)
        print("引用的知识库片段:\n", retrieved_chunks)
        reranked_chunks = rerank(prompt, retrieved_chunks, top_k, cross_encoder)
        joined_chunks = "\n".join(reranked_chunks)
        prompt = f"""日志摘要:{prompt}\n\n先验知识:{joined_chunks}"""   
    system_prompt = "你是一个电子设备日志分析专家，请根据日志信息，给出最简单精炼日志摘要，并给出具体的建议执行操作"
    #full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.85,
        top_k=30,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]  # 只拿生成的新token
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    end_time = time.time()
    print("大模型运行时间:", end_time - start_time)

    return generated_text

def model_initial(
        llm_path,
        embedding_model,
        cross_encoder,
        chromadb_path,
        soh_path, 
        cls_path,
        normal_features=None,
        threshold=0.05, 
        num_modules=10,
        normalize=True,
        filter=True,
        auto_threshold=False,
        rag_enable=False,
        seed=42,
        enable_llm=False,
        *args,
        **kwargs
):
    """
    加载模型，返回各模型对象
    """
    if seed is not None:
        set_random_seed(seed)
   
    # Get device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA (GPU {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no MPS or CUDA found)")
    dtype = torch.float32  # 使用一致的数据类型

    soh_detector = None
    # 加载大模型
    torch.cuda.empty_cache()
    print(f"Loading LLM from {llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True).to(device)
        
    # 加载RAG模型
    if rag_enable:
        print(f"Loading Embedding Model from {embedding_model}")
        embedding_model = SentenceTransformer(embedding_model)
        print(f"Loading Chromadb Collection from {chromadb_path}")
        chromadb_client = chromadb.PersistentClient(chromadb_path)
        chromadb_collection = chromadb_client.get_or_create_collection(name="default")
        print(f"Loading Cross-Encoder from {cross_encoder}")
        cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    # 加载SOH模型权重
    print('Loading SOH Model...')
    checkpoint = torch.load(soh_path, map_location=device)
    soh_predictor = SOHPredictor(input_dim=normal_features.shape[1]).to(device)
    soh_predictor.load_state_dict(checkpoint)

    checkpoint = torch.load(cls_path, map_location=device)
    cls_model = ClassificationModel(input_dim=normal_features.shape[1], num_modules=num_modules).to(device)
    cls_model.load_state_dict(checkpoint)
        
    print(f"✅Loading Model Finished!")

    soh_detector = SOHDetector(
            soh_predictor, 
            cls_model, 
            normal_features, 
            device, 
            threshold=threshold, 
            auto_threshold=auto_threshold, 
            normalize=normalize,
            sclar_soh_path='./outputs/scaler_soh.pkl',
            sclar_cls_path='./outputs/scaler_cls.pkl',
            filter=filter,
            print_log=False,
    )
        
    return {
        'device': device,
        'soh_detector': soh_detector,
        'llm_model': llm_model,
        'tokenizer': tokenizer,
        'embedding_model': embedding_model,
        'cross_encoder': cross_encoder,
        'chromadb_collection': chromadb_collection
    }

def on_receive(transfer: RealTimeTransfer, info: Union[dict, str], results_history: dict, model_dict: dict, enable_llm: bool, device="cpu") -> dict:
    text: str = None
    # 接收健康度信息
    if not enable_llm:
        for key, value in zip(["soh", "threshold", "warning", "rul", "log"], info):
            results_history[key].append(value)
            # 超出长度
            if len(results_history["soh"]) > 128:
                for key in ["soh", "threshold", "warning", "rul", "log"]:
                    results_history[key].pop(0)
            # 调用绘图
            print(f"健康度:{results_history['soh'][-1]}")
            #realtime_soh_plot(results_history, 128)
    else:
        # 调用大模型处理
        print(f"接收信息: {info}")
        # 将大模型处理放入队列，立即返回（不阻塞）
        model_queue.put((info, transfer, results_history))
        text = receive_info["response"][-1]
        
    # 写json文件
    receive_info = {
        "soh": results_history["soh"][-1],
        "threshold": results_history["threshold"][-1],
        "warning": results_history["warning"][-1],
        "rul": results_history["rul"][-1],
        "log": results_history["log"][-1],
        "response": text if text else ""
    }
    soh_buffer = [receive_info["soh"], receive_info["threshold"], receive_info["warning"], receive_info["rul"], receive_info["log"]]
    #send_soh(transfer, soh_buffer)
    send_soh_async(soh_buffer)
    if receive_info["response"]:
        #send_message(transfer, receive_info["response"])
        send_message_async(receive_info["response"])
    # with open('./outputs/results.json', 'a') as f:
    #     json.dump(receive_info, f, ensure_ascii=False, indent=4)

    return copy.deepcopy(info)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default='Qwen/Qwen1.5-1.8B-Chat')
    parser.add_argument("--embedding_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--cross_encoder", type=str, default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--chromadb_path", type=str, default="./dataset/doc.db")
    parser.add_argument("--soh_path", type=str, default='./outputs/best_sohmodel.pth')
    parser.add_argument("--cls_path", type=str, default='./outputs/best_classification.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/无异常.txt')
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--num_normal_samples", type=int, default=32, help="多少个正常样本用于求解正常时的loss")
    parser.add_argument("--num_detect_samples", type=int, default=32, help="以多少个样本为一组进行预测，提高鲁棒性")
    parser.add_argument("--num_modules", type=int, default=10, help="模块数")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--filter", type=bool, default=True, help="是否进行输出滤波")
    parser.add_argument("--auto_threshold", type=bool, default=True, help="自适应阈值")
    parser.add_argument("--rag_enable", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)  #999
    parser.add_argument("--my_ip", type=str, default="192.168.2.1")
    parser.add_argument("--my_port", type=int, default=8888)
    parser.add_argument("--peer_uri", type=str, default="ws://localhost:3016/ws/receive")
    parser.add_argument("--enable_llm", type=bool, default=False)

    args = parser.parse_args()
    return args

def main(args, **kwargs):
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    data_list, nlen, nlen2 = data_process(args.data_path) # 前nlen个样本是bad data
    
    # 处理数据  将data_list从字典格式转化为np.array，并使用np.vstack将其堆叠成一个二维数组
    features = []
    for entry in data_list:
        feature_vector = np.array(list(entry.values()))
        features.append(feature_vector)
    features = np.vstack(features)
    print(f'Feature Shape:{features.shape}')
    # 先根据正常工作时的样本得到正常时的loss大小
    normal_features = np.concatenate((features[:nlen], features[nlen + nlen2:]), axis=0)
    # 从正常样本中随机选取num_detect_samples个样本作为正常样本
    inx = np.random.randint(0, nlen-args.num_normal_samples)
    #normal_features = normal_features[inx:inx + num_normal_samples]
    normal_features = normal_features[np.random.choice(len(normal_features), args.num_normal_samples, replace=False)]
    print(f'😋Using Normal Samples: {len(normal_features)}')
    print(f'Normal Samples: {normal_features.shape}')

    model_dict = model_initial(**vars(args), normal_features=normal_features)
    soh_predicter = RealTimeSOHPredicter(model_dict, args.num_detect_samples, None)
    data_reader = RealTimeDataReader(args.data_path, args.add_noise)

    # 启动Websocket
    transfer = RealTimeTransfer(
        my_ip=None,
        my_port=None,
        peer_uri="ws://localhost:3016/ws/receive"
    )

    transfer.log_path = "./outputs/log_transfer.txt"
    transfer.write_log("✅ 正在启动新的client(标识ID:LZH-DEBUG)", append=False)
    transfer.run(mode="client")
    print("Client started!!")

    # 启动发送线程
    sender = threading.Thread(target=sender_thread, args=(transfer, send_queue), daemon=True)
    sender.start()

    # 模型开始运行
    send_cnt = 128
    results_history = {
        "soh": [],
        "threshold": [],
        "warning": [],
        "rul": [],
        "log": [],
        "response": [],
    }
    
    # 清空日志内容
    soh_predicter.soh_detector.write_log("BEGIN", append=False)
    while transfer._running == False:
        time.sleep(1)
        print("等待连接...")

    # 循环读取数据
    print("连接成功, 模型开始运行...")
    while row_data := data_reader.read_next():
        row_feature = np.array(list(row_data.values()))
        result_dict = soh_predicter.add_sample(row_feature)
        #realtime_soh_plot(soh_predicter.results, soh_predicter.remembered_max_result)

        # 发送健康度和阈值给服务器保存
        if result_dict is not None:
            last_msg = on_receive(transfer=transfer,
                                  info=result_dict,
                                  results_history=results_history,
                                  model_dict=model_dict,
                                  enable_llm=False,
                                  device=model_dict["device"]
                                )
        # 发送压缩日志信息给服务器大模型推理
        if len(soh_predicter.soh_detector.log_info) % send_cnt == 0 and len(soh_predicter.soh_detector.log_info) > 0:
            # 将压缩日志发送给服务器端
            prompt = compress_logs_for_model(soh_predicter.soh_detector.log_info[-send_cnt:])
            last_msg = on_receive(transfer=transfer,
                                  info=prompt,
                                  results_history=results_history,
                                  model_dict=model_dict,
                                  enable_llm=True,
                                  device=model_dict["device"]
                                )
            
        time.sleep(0.2)


if __name__ == '__main__':
    args = parse_args()
    main(args)