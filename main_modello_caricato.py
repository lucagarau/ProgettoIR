import torch
import torch.optim as optim
import numpy as np
import time
import random
import os
import argparse
from dataloader import GroupDataset
from metrics import evaluate2
from model import DHMAE
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def training(model, optimizer, train_loader, type_m):
    losses = []
    if type_m == "user":
        for _, (u, pi_ni) in enumerate(train_loader):
            user_input = u.to(device)
            pos_items_input, neg_items_input = pi_ni[:, 0].to(device), pi_ni[:, 1].to(
                device
            )
            optimizer.zero_grad()
            loss = model.bpr_loss(user_input, pos_items_input, neg_items_input)
            losses.append(loss)
            loss.backward()
            optimizer.step()
    elif type_m == "group":
        for _, (g, pi_ni) in enumerate(train_loader):
            group_input = g.to(device)
            pos_items_input, neg_items_input = pi_ni[:, 0].to(device), pi_ni[:, 1].to(
                device
            )
            optimizer.zero_grad()
            loss_ae, nodes_x = model.ae_loss()
            loss = (
                model.bpr_loss(
                    group_input, pos_items_input, neg_items_input, nodes_x, "group"
                )
                + loss_ae
            )
            losses.append(loss)
            loss.backward()
            optimizer.step()

    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    # load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        help="[CAMRa2011, Mafengwo, MafengwoS, MovieLens, WeeplacesS]",
        default="Mafengwo",
    )
    parser.add_argument(
        "--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0"
    )
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--emb_dim", type=int, default=32)

    parser.add_argument("--num_enc_layers", type=int, default=3)
    parser.add_argument("--num_dec_layers", type=int, default=2)
    parser.add_argument("--sce_alpha", type=int, default=1)
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--num_negatives", type=int, default=12)
    parser.add_argument("--epoch", type=int, default=30)

    parser.add_argument("--topK", type=list, default=[1, 5, 10, 20, 50])

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(vars(args))
    print("= = = = = = = = = = = = = = = = = = = =")
    # load dataset
    dataset = GroupDataset(dataset=args.dataset)
    # create model
    train_model = DHMAE(
        dataset.num_users,
        dataset.num_items,
        dataset.num_groups,
        dataset.group_member_dict,
        {
            "user_hg": dataset.user_hyper_graph.to(device),
            "item_hg": dataset.item_hyper_graph.to(device),
            "full_hg": dataset.full_hg.to(device),
        },
        dataset.mask_rate_mat.to(device),
        args,
    ).to(device)
    # optimizer = optim.Adam(train_model.parameters(), lr=args.learning_rate)
    # save_path = f"saved_models/{args.dataset}"
    # os.makedirs(save_path, exist_ok=True)
    # '''
    # # train
    # for epoch_id in range(1, args.epoch + 1):
    #     train_model.train()
    #     user_loss, group_loss = 0, 0
    #     st_time = time.time()
    #     user_loss = training(
    #         train_model,
    #         optimizer,
    #         dataset.get_user_train_dataloader(args.batch_size, args.num_negatives),
    #         "user",
    #     )
    #     group_loss = training(
    #         train_model,
    #         optimizer,
    #         dataset.get_group_train_dataloader(args.batch_size, args.num_negatives),
    #         "group",
    #     )
    #
    #     print(
    #         f"Epoch {epoch_id}: Cost time: {time.time() - st_time:4.2f}s, Loss: [User->{user_loss:.7f}, Group->{group_loss:.7f}]"
    #     )
    #
    # model_save_file = os.path.join(save_path, "model_last.pth")
    # torch.save(train_model.state_dict(), model_save_file)
    # print(f"Model saved in in: {model_save_file}")
    # print("= = = = = = = = = = = = = = = = = = = =")
    # # test
    # '''
    print("modello in caricamento")
    train_model.load_state_dict(torch.load(f"./saved_models/{args.dataset}/model_last.pth",weights_only=True))
    print("modello caricato")
    train_model.eval()
    # user_hrs, user_ndcgs, user_mrr,user_hits5, user_hitsless5, user_ndcg_pop, user_ndcg_npop = evaluate2(
    #     train_model,
    #     dataset.user_test_ratings,
    #     dataset.user_test_negatives,
    #     device,
    #     args.topK,
    #     "user",
    # )
    user_hrs, user_ndcgs, user_mrr, user_hits5, user_hitsless5, user_ndcg_pop, user_ndcg_npop, user_mrp_pop, user_mrp_npop = evaluate2(
        train_model,
        dataset.user_test_ratings,
        dataset.user_test_negatives,
        device,
        args.topK,
        "user",
    )

    group_hrs, group_ndcgs, group_mrr,group_hits5, group_hitsless5, group_ndcg_pop, group_ndcg_npop, group_mrp_pop, group_mrp_npop = evaluate2(
        train_model,
        dataset.group_test_ratings,
        dataset.group_test_negatives,
        device,
        args.topK,
        "group",
    )
    print(f"User->HR@{args.topK}: {user_hrs}, \n"
          f"NDCG@{args.topK}: {user_ndcgs}, \n"
          f"MRR@{args.topK}: {user_mrr}, \n"
          f"hits_K_gt5{args.topK}: {user_hits5}, \n"
          f"hits_K_lt5: {user_hitsless5}, \n"
          f"ndcg_pop{args.topK}: {user_ndcg_pop}, \n"
          f"ndcg_npop: {user_ndcg_npop}, \n"
          f"mrp_K_gt5{args.topK}: {user_mrp_pop}, \n"
          f"mrp_K_lt5: {user_mrp_npop}")

    print(f"Group->HR@{args.topK}: {group_hrs}, \n"
          f"NDCG@{args.topK}: {group_ndcgs}, \n"
          f"MRR@{args.topK}: {group_mrr}, \n"
          f"hits_K_gt5{args.topK}: {group_hits5}, \n"
          f"hits_K_lt5{args.topK}: {group_hitsless5}, \n"
          f"ndcg_pop{args.topK}: {group_ndcg_pop}, \n"
          f"ndcg_npop{args.topK}: {group_ndcg_npop}, \n"
          f"mrp_K_gt5{args.topK}: {group_mrp_pop}, \n"
          f"mrp_K_lt5{args.topK}: {group_mrp_npop}")
    print("Done!")
