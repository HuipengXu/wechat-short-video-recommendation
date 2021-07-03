import os
import gc
import numpy as np
from comet_ml import Experiment
from tqdm import tqdm, trange
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel)

from .utils import (
    seed_everything, roc_score, load_embedding_matrix, EMA, ProcessedData,
    batch2cuda, WXDataset, Collator, roc_score_one_act, reset_args, undersampling_neg_sample, read_data,
    undersampling_neg_sample_all_actions
)

import torch
import torch.nn as nn
from torch.optim import Adagrad,AdamW
from torch.utils.data import DataLoader

from .multideepfm4wx import MultiDeepFM
# from .transformer4wx import TransformerForWX
from .args import get_args

ACTIONS = ['read_comment', 'like', 'click_avatar', 'forward']


def comet_setup():
    experiment = Experiment(
        api_key="spdTp4F6yw6vFafHBN4uejAQA",
        project_name="WX-CTR-REAL",
        workspace="tangzhihong",
        log_code=True,
        log_graph=True,
        parse_args=True,
        disabled=False
    )
    experiment.log_asset_folder(os.path.dirname(__file__), log_file_name=False)
    return experiment


EXPERIMENT = comet_setup()


def evaluation(config, model, val_dataloader):
    model.eval()
    preds = []
    nums_pred_list = []
    labels = []
    user_ids = []
    val_loss = 0.
    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda, user_id = batch2cuda(batch, config)
            user_ids.extend(user_id)
            pos_loss, neg_loss, loss, nums_loss, logits, label, nums_logits = model(**batch_cuda)
            labels.append(label.detach().cpu())
            nums_preds = (F.sigmoid(nums_logits) > 0.5).int()
            nums_pred_list.append(nums_preds.detach().cpu())
            probs = torch.sigmoid(logits)

            loss /= len(user_id)
            if config.n_gpus > 1:
                loss = loss.mean()

            val_loss += loss.item()
            preds.append(probs.detach().cpu())
    total_labels = torch.cat(labels, 0)
    total_nums_pred = torch.cat(nums_pred_list, 0)
    acc = torch.sum((total_labels.sum(1, keepdim=True) > 0).int() ==total_nums_pred)/(len(total_nums_pred)*1.0)
    avg_val_loss = val_loss / len(val_dataloader)
    if config.action:
        metrics = roc_score_one_act(labels, preds, user_ids, config.action)
    else:
        metrics = roc_score(labels, preds, user_ids)
    metrics[f'{config.action}_val_loss'] = avg_val_loss
    metrics[f'num_acc'] = acc.item()
    return metrics


def train(config, train_dataloader, valid_dataloader):
    # model = DeepFM(config)
    model = MultiDeepFM(config)
    model.to(config.device)
    # model = TransformerForWX(config)
    # optimizer = AdamW(model.parameters(), lr=config.lr, eps=1.0e-4, weight_decay=0.001)
    optimizer = Adagrad(model.parameters(), lr=config.lr)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataloader)*config.num_epochs*0.2), num_training_steps=int(len(train_dataloader)*config.num_epochs))
    # optimizer = Lookahead(optimizer, 5, 1)
    # total_steps = config['num_epochs'] * len(train_dataloader)
    epoch_iterator = trange(config.num_epochs, desc='Epoch')
    global_steps = 0
    train_loss = 0.
    pos_train_loss = 0.
    train_num_loss = 0.
    neg_train_loss = 0.
    logging_loss = 0.
    best_roc_auc = 0.
    best_model_path = ''

    if config.n_gpus > 1:
        model = nn.DataParallel(model)

    optimizer.zero_grad()

    for _ in epoch_iterator:

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch_cuda, user_ids = batch2cuda(batch, config)
            total_reg_loss = 0.
            pos_loss, neg_loss, loss, nums_loss = model(**batch_cuda)[:4]
            for n, p in model.named_parameters():
                if "multi_modal_embedding" in n:
                    continue
                if "weight" in n:
                    if "embedding" in n:
                        total_reg_loss += torch.sum(config.l2_reg_embedding * torch.square(p))
                    else:
                        total_reg_loss += torch.sum(config.l2 * torch.square(p))
            loss += total_reg_loss
            loss /= len(user_ids)
            if config.n_gpus > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            if config.ema_start:
                ema.update()

            train_loss += loss.item()
            pos_train_loss += pos_loss.item()
            neg_train_loss += neg_loss.item()
            train_num_loss += nums_loss.item()
            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            # EXPERIMENT.log_metric(f'running {config.action} training loss', loss.item(), step=global_steps)
            if global_steps % config.logging_steps == 0:
                if global_steps >= config.ema_start_step and not config.ema_start:
                    print('\n>>> EMA starting ...')
                    config.ema_start = True
                    ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)

                print_train_loss = (train_loss - logging_loss) / config.logging_steps
                print_pos_loss = pos_train_loss / config.logging_steps
                print_neg_loss = neg_train_loss / config.logging_steps
                print_num_loss = train_num_loss / config.logging_steps
                logging_loss = train_loss
                pos_train_loss = 0.
                neg_train_loss = 0.
                train_num_loss = 0.
                if config.ema_start:
                    ema.apply_shadow()
                best_model_path, best_roc_auc = evaluation_and_save(best_model_path, best_roc_auc, config,
                                                                    global_steps, model,
                                                                    print_train_loss, print_pos_loss, print_neg_loss, print_num_loss, valid_dataloader)
                model.train()
                if config.ema_start:
                    ema.restore()

    if global_steps % config.logging_steps != 0:
        print_train_loss = (train_loss - logging_loss) / (global_steps % config.logging_steps)
        print_pos_loss = pos_train_loss / (global_steps % config.logging_steps)
        print_neg_loss = neg_train_loss / (global_steps % config.logging_steps)
        print_num_loss = train_num_loss / (global_steps % config.logging_steps)
        best_model_path, best_roc_auc = evaluation_and_save(best_model_path, best_roc_auc, config,
                                                            global_steps, model,
                                                            print_train_loss, print_pos_loss, print_neg_loss, print_num_loss, valid_dataloader)

    return model, best_model_path


def evaluation_and_save(best_model_path, best_roc_auc, config, global_steps, model,
                        print_train_loss, print_pos_loss, print_neg_loss, print_num_loss, valid_dataloader):
    metrics = evaluation(config, model, valid_dataloader)
    roc_auc = metrics['avg_roc'] if not config.action else metrics[config.action]
    print_log = f'\n>>> training loss: {print_train_loss:.4f} '
    if roc_auc > best_roc_auc:
        model_save_path = os.path.join(config.output_dir,
                                       f'checkpoint-{global_steps}-{roc_auc:.6f}.pth')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), model_save_path)
        torch.save(model_to_save.state_dict(), os.path.join(config.output_dir,"best_model.pth"))
        best_roc_auc = roc_auc
        best_model_path = model_save_path
    for metric, score in metrics.items():
        print_log += f'{metric}: {score:.4f} '
    print(print_log)
    metrics[f'{config.action}_train_loss'] = print_train_loss
    metrics[f'{config.action}_pos_loss'] = print_pos_loss
    metrics[f'{config.action}_neg_loss'] = print_neg_loss
    metrics[f'{config.action}_num_loss'] = print_num_loss
    EXPERIMENT.log_metrics(metrics, step=global_steps)
    return best_model_path, best_roc_auc


def merge_multi_act_res(config):
    test_df = pd.read_csv(config.submit_test_path).drop('device', axis=1)
    read_comment_df = pd.read_csv(os.path.join(config.output_dir, 'read_comment', 'submission.csv'))
    like_df = pd.read_csv(os.path.join(config.output_dir, 'like', 'submission.csv'))
    click_avatar_df = pd.read_csv(os.path.join(config.output_dir, 'click_avatar', 'submission.csv'))
    forward_df = pd.read_csv(os.path.join(config.output_dir, 'forward', 'submission.csv'))
    sub_df = pd.concat([test_df, read_comment_df.read_comment,
                        like_df.like, click_avatar_df.click_avatar,
                        forward_df.forward], axis=1)
    submission_path = os.path.join(config.output_dir, 'merge_submission.csv')
    sub_df.to_csv(submission_path, index=False, encoding='utf8', sep=',')


# def prepare_data(args):
#     data_df = read_data(args.merge_data_path, args.debug_data)
#     test_df = read_data(args.test_data_path, args.debug_data)
#     processed_data = ProcessedData(args)
#     test_dataset = WXDataset(test_df, processed_data)
#     collate_fn = Collator(args)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
#                                  collate_fn=collate_fn, num_workers=4, pin_memory=True)
#     return collate_fn, data_df, test_dataloader, processed_data


def prepare_data(args):
    data_df = read_data(args.merge_data_path, args.debug_data)
    processed_data = ProcessedData(args)
    collate_fn = Collator(args)
    return collate_fn, data_df, processed_data


def args_setup():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.n_gpus = torch.cuda.device_count()
        args.bs *= args.n_gpus
    args.multi_modal_emb_matrix = load_embedding_matrix(
        filepath=args.multi_modal_emb_matrix_path,
        max_vocab_size=args.vocab_size
    )
    args.multi_modal_emb_char_matrix = load_embedding_matrix(
        filepath=args.multi_modal_emb_char_matrix_path,
        max_vocab_size=args.char_vocab_size
    )
    return args


def build_dataloader(processed_data, data_df, args, collate_fn):
    # train_df = data_df.loc[data_df.date_ != 14]
    # valid_df = data_df.loc[data_df.date_ == 14]
    train_size = int(len(data_df) * 0.8)
    train_df = data_df.iloc[:train_size]
    valid_df = data_df.iloc[train_size:]
    train_dataset = WXDataset(train_df, processed_data)
    valid_dataset = WXDataset(valid_df, processed_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                  collate_fn=collate_fn, num_workers=6, pin_memory=False)
    # train_sampler = RandomSampler(train_dataset)
    # bucket_sampler = BucketBatchSampler(train_sampler, batch_size=args.bs, drop_last=False,
    #                                     sort_key=lambda x: max(len(train_dataset[x][i]) for i in range(-7, -1)),
    #                                     bucket_size_multiplier=args.bucket_size_multiplier)
    # train_dataloader = DataLoader(train_dataset, batch_sampler=bucket_sampler, num_workers=4,
    #                               collate_fn=collate_fn, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=False,
                                  collate_fn=collate_fn, num_workers=6, pin_memory=True)
    return train_dataloader, valid_dataloader


def main():
    args = args_setup()

    collate_fn, data_df, processed_data = prepare_data(args)
    EXPERIMENT.log_parameters(args)

    if args.action:
        for action in ACTIONS:
            print(f'\n>>> Starting training {action} model ...')
            reset_args(action, args)
            action_df, test_df = undersampling_neg_sample(data_df, args)
            test_dataloader = DataLoader(WXDataset(test_df, processed_data), batch_size=args.bs, shuffle=False,
                                         collate_fn=collate_fn, num_workers=4, pin_memory=True)
            train_dataloader, valid_dataloader = build_dataloader(processed_data, action_df, args, collate_fn)
            model, args.best_model_path = train(args, train_dataloader, valid_dataloader)
            del model, train_dataloader, valid_dataloader
            gc.collect()
            torch.cuda.empty_cache()
        print('\n>>> Starting merge all actions results ...')
        merge_multi_act_res(args)
    else:
        action_list = ["read_comment", "like", "click_avatar", "forward"]
        os.makedirs(args.output_dir, exist_ok=True)
        test_df = read_data(args.test_data_path, args.debug_data)
        mms = MinMaxScaler(feature_range=(0, 1))
        data_df[['videoplayseconds']] = mms.fit_transform(data_df[['videoplayseconds']])
        test_df[['videoplayseconds']] = mms.transform(test_df[['videoplayseconds']])
        
        merge_all_data = pd.concat([data_df, test_df], axis=0, ignore_index=True)
        merge_all_data["user_seq"] = merge_all_data.groupby(["userid"]).cumcount() + 1

        data_df = merge_all_data[~merge_all_data['read_comment'].isna()].reset_index(drop=True)
        test_df = merge_all_data[merge_all_data['read_comment'].isna()].reset_index(drop=True)
        test_df = test_df.dropna(axis=1, how="all")
        del merge_all_data

        data_df["action_num"] = data_df[action_list].sum(axis=1)
        sample_data_df = undersampling_neg_sample_all_actions(data_df, args)
        # manual collect the rubbish
        del data_df
        gc.collect()
        train_dataloader, valid_dataloader = build_dataloader(processed_data, sample_data_df, args, collate_fn)
        model, args.best_model_path = train(args, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
