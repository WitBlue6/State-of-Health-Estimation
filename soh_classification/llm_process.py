import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from transformers import AutoModelForCausalLM, AutoTokenizer
#from fastapi import FastAPI
import torch
import torch.nn as nn
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from model import set_random_seed
from utils import *
from websocket_transfer import RealTimeTransfer
from rag import retrieve, rerank
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import json
import queue
import threading

def receiver_thread(receiver: RealTimeTransfer, msg_queue: queue.Queue):
    '''独立线程, 用于接收websocket消息'''
    while True:
        if receiver.received:
            msg_queue.put(receiver.receive_info)
            receiver.received = False
        time.sleep(0.01)

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
        annotation_text = f"SOH = {results['soh'][-1]:.2f}\nRUL = {results['rul'][-1]:.2f}"
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

    if enable_llm:
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
    else:
        tokenizer = None
        llm_model = None
        embedding_model = None
        chromadb_collection = None
        cross_encoder = None
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

def on_receive(info: dict, results_history: dict, model_dict: dict, last_msg=None, device="cpu") -> dict:
    text: str = None
    if info["type"] == "list" and last_msg != info:  # 接收健康度信息
        for key, value in zip(["soh", "threshold", "warning", "rul", "log"], info["data"]):
            results_history[key].append(value)
        # 超出长度
        if len(results_history["soh"]) > 128:
            for key in ["soh", "threshold", "warning", "rul", "log"]:
                results_history[key].pop(0)
        # 调用绘图
        print(f"健康度:{results_history['soh'][-1]}")
        #realtime_soh_plot(results_history, 128)
    elif info["type"] == "text" and last_msg != info:  # 接收日志信息
        input = info["data"]
        results_history["response"].append(input)
        print(f"接收信息: {input}")
        # 调用大模型处理
        text = generate_text(
            model=model_dict["llm_model"],
            tokenizer=model_dict["tokenizer"],
            prompt=input,
            rag=True,
            embedding_model=model_dict["embedding_model"],
            chromadb_collection=model_dict["chromadb_collection"],
            cross_encoder=model_dict["cross_encoder"],
            device=device
        )
        print(f"大模型输出:\n{text}")
    # 写json文件
    receive_info = {
        "soh": results_history["soh"][-1],
        "threshold": results_history["threshold"][-1],
        "warning": results_history["warning"][-1],
        "rul": results_history["rul"][-1],
        "log": results_history["log"][-1],
        "response": text if text else ""
    }
    with open('./outputs/results.json', 'a') as f:
        json.dump(receive_info, f, ensure_ascii=False, indent=4)

    return copy.deepcopy(info)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default='Qwen/Qwen1.5-1.8B-Chat')
    parser.add_argument("--embedding_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--cross_encoder", type=str, default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--chromadb_path", type=str, default="./dataset/doc.db")
    parser.add_argument("--soh_path", type=str, default='./outputs/best_sohmodel.pth')
    parser.add_argument("--cls_path", type=str, default='./outputs/best_classification.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/舵机1故障.txt')
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
    parser.add_argument("--my_ip", type=str, default="localhost")
    parser.add_argument("--my_port", type=int, default=8765)
    parser.add_argument("--peer_uri", type=str, default="ws://10.63.6.105:8765")

    args = parser.parse_args()


    model_dict = model_initial(**vars(args), enable_llm=True)
    
    # 启动Websocket
    receiver = RealTimeTransfer(
        my_ip=args.my_ip,
        my_port=args.my_port,
        peer_uri=args.peer_uri,
    )
    receiver.log_path = "./outputs/log_receiver.txt"
    receiver.write_log("✅ 正在启动新的server(标识ID:LZH-DEBUG)", append=False)
    receiver.run(mode="server")
    print("Server started!!")
    results_history = {
        "soh": [],
        "threshold": [],
        "warning": [],
        "rul": [],
        "log": [],
        "response": [],
    }

    #receicer.on_receive_callback = lambda info: on_receive(info, results_history, model_dict)
    # 启动接收线程
    msg_queue = queue.Queue()
    receiver_thread = threading.Thread(target=receiver_thread, args=(receiver, msg_queue), daemon=True)
    receiver_thread.start()
    # 取队列消息
    last_msg = None
    while True:
        try:
            info = msg_queue.get()
            #print("DEBUG:", info)
            last_msg = on_receive(info=receiver.receive_info,
                                  results_history=results_history,
                                  model_dict=model_dict,
                                  last_msg=last_msg,
                                  device=model_dict["device"]
                                )
        except queue.Empty:
            pass
