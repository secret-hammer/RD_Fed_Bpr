import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset
import xlwt

from FedRec.server import FedRecServer
from FedRec.adapt_client import FedRecClient


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)
    dims = [32, 64, 128, 256]
    dim_num = dict.fromkeys(dims, 0)

    t0 = time()
    m_item, all_train_ind, all_test_ind, part_train_ind = load_dataset(args.path + args.dataset)

    server = FedRecServer(m_item, args.dim).to(args.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        dim = random.choice(dims)
        dim_num[dim] += 1
        clients.append(
            FedRecClient(train_ind, test_ind, m_item, dim, args.dim).to(args.device)
        )

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % \
          (time() - t0, len(clients), m_item, sum([len(i) for i in all_train_ind]), sum([len(i) for i in all_test_ind])))
    print("dim--clients: 32 -- %d 64 -- %d 128 -- %d 256 -- %d" % (dim_num[32], dim_num[64], dim_num[128], dim_num[256]))
    print("output format: ({Sampled HR@10}), ({ER@5},{ER@10},{NDCG@10})")

    # Init performance
    t1 = time()
    with torch.no_grad():
        test_result = server.eval_(clients)
    print("Epoch 0(init), (%.4f) on test" % tuple(test_result) +
          " [%.1fs]" % (time() - t1))

    wb = xlwt.Workbook()
    table = wb.add_sheet('test_results')
    table_head = ['Epoch', 'Loss', 'PA']
    for i in range(len(table_head)):
        table.write(0, i, table_head[i])

    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            with torch.no_grad():
                test_result = server.eval_(clients)

            table.write(epoch, 0, 'Epoch ' + str(epoch))
            table.write(epoch, 1, total_loss)
            table.write(epoch, 2, (tuple(test_result))[0].item())

            if epoch % 10 == 0:
                print("Epoch %d, loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1) +
                      ", (%.4f) on test" % tuple(test_result) +
                      " [%.1fs]" % (time() - t2))
        wb.save('Data/test_result/test_adapt.xls')
    except KeyboardInterrupt:
        pass



setup_seed(20220403)

if __name__ == "__main__":
    main()
