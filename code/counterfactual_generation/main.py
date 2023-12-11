import tqdm
import torch
import random
import argparse
import numpy as np
from torch.optim import AdamW
from preprocess import DatasetLoader
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def train(args):
    random.seed(args.seed)                                               
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    dataset_loader = DatasetLoader(args)
    # config = AutoConfig.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = dataset_loader.get_trainset()
    dataset = TensorDataset(input_ids)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    model.train()
    for _ in tqdm.tqdm(range(args.epoch)):
        for input_ids in loader:
            input_ids = input_ids[0]
            input_ids = input_ids.to(device)
            outputs = model.forward(input_ids, labels=input_ids)

            loss = outputs.loss
            loss = loss.mean()
            print("Current loss is:", loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(args.saved_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ace2005", type=str)
    parser.add_argument("--model_dir", default="gpt2-base", type=str)
    parser.add_argument("--saved_dir", default="saved_model", type=str)
    parser.add_argument("--max_length", default=47, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    current_args = parser.parse_args()

    train(current_args)
