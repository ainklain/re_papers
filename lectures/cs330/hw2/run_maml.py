
"""
python -m hw2.run_maml.py

Usage Instructions:
    5-way, 1-shot omniglot:
        python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
    20-way, 1-shot omniglot:
        python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
    To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
"""
import argparse
import csv
import numpy as np
import os
import pickle
import random
import torch
from tensorboardX import SummaryWriter

from .load_data import DataGenerator
from .models.maml import MAML


parser = argparse.ArgumentParser(description='MAML')

## Dataset/method options
parser.add_argument('--n_way', default=5, type=int, help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--meta_train_iterations', default=15000, type=int, help='number of meta-training iterations.')
parser.add_argument('--meta_batch_size', default=25, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--k_shot', default=1, type=int, help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--inner_update_lr', default=0.4, type=float, help='step size alpha for inner gradient update.')
parser.add_argument('--num_inner_updates', default=1, type=int, help='number of inner gradient updates during meta-training.')
parser.add_argument('--num_filters', default=16, type=int, help='number of filters for conv nets.')
parser.add_argument('--learn_inner_update_lr', default=False, type=bool, help='learn the per-layer update learning rate.')

## Logging, saving, and testing options
parser.add_argument('--data_path', default='./data/omniglot_resized', type=str, help='path to the dataset.')
parser.add_argument('--log', default=True, type=bool, help='if false, do not log summaries, for debugging code.')
parser.add_argument('--logdir', default='/tmp/data', type=str, help='directory for summaries and checkpoints.')
parser.add_argument('--resume', default=False, type=bool, help='resume training if there is a model available')
parser.add_argument('--meta_train', default=True, type=bool, help='True to meta-train, False to meta-test.')
parser.add_argument('--meta_test_iter', default=-1, type=int, help='iteration to load model (-1 for latest model)')
parser.add_argument('--meta_test_set', default=False, type=bool, help='Set to true to test on the the meta-test set, False for the meta-training set.')
parser.add_argument('--meta_train_k_shot', default=-1, type=int, help='number of examples used for gradient update during meta-training (use if you want to meta-test with a different number).')
parser.add_argument('--meta_train_inner_update_lr', default=-1, type=float, help='value of inner gradient step step during meta-training. (use if you want to meta-test with a different value)')
parser.add_argument('--meta_test_num_inner_updates', default=1, type=int, help='number of inner gradient updates during meta-test.')

args = parser.parse_args()


class Saver:
    def save(path, ep, model, optimizer):
        save_path = os.path.join(path, "saved_model.pt")
        torch.save({
            'ep': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print('model saved successfully. ({})'.format(path))


    def load(path, model, optimizer):
        load_path = os.path.join(path, "saved_model.pt")
        if not os.path.exists(load_path):
            return False

        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(tu.device)
        model.eval()
        print('model loaded successfully. ({})'.format(path))
        return checkpoint['ep']


def meta_train(args, model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 10    # interval for writing a summary (reduced from 100)
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10      # interval for how often to print (reduced from 100)
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if args.log:
        train_writer = SummaryWriter()
        train_writer.add_graph(model)
    print('Done initializing, starting training.')
    pre_accuracies, post_accuracies = [], []

    num_classes = data_generator.num_classes

    for itr in range(resume_itr, args.meta_train_iterations):
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of training data and partition into
        # group a (inputa, labela) and group b (inputb, labelb)

        # inputa, inputb, labela, labelb = None, None, None, None
        # #############################
        # feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
        #
        # input_tensors = [model.metatrain_op]
        #
        # if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
        #     input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[args.num_inner_updates-1],
        #                             model.total_accuracy1, model.total_accuracies2[args.num_inner_updates-1]])
        #
        # result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            pre_accuracies.append(result[-2])
            if args.log:
                train_writer.add_scalar('total_loss1', result[1], itr)
                # train_writer.add_summary(result[1], itr)
            post_accuracies.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration %d: pre-inner-loop accuracy: %.5f, post-inner-loop accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
            print(print_str)
            pre_accuracies, post_accuracies = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(args.logdir + '/' + exp_string + '/model' + str(itr))
            # saver.save(sess, args.logdir + '/' + exp_string + '/model' + str(itr))

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            #############################
            #### YOUR CODE GOES HERE ####

            # sample a batch of validation data and partition into
            # group a (inputa, labela) and group b (inputb, labelb)

            inputa, inputb, labela, labelb = None, None, None, None
            #############################
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
            input_tensors = [model.total_accuracy1, model.total_accuracies2[args.num_inner_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Meta-validation pre-inner-loop accuracy: %.5f, meta-validation post-inner-loop accuracy: %.5f' % (result[-2], result[-1]))

    # saver.save(sess, args.logdir + '/' + exp_string + '/model' + str(itr))
    saver.save(args.logdir + '/' + exp_string + '/model' + str(itr))

# # calculated for omniglot
# NUM_META_TEST_POINTS = 600
#
# def meta_test(model, saver, sess, exp_string, data_generator, meta_test_num_inner_updates=None):
#     num_classes = data_generator.num_classes
#
#     np.random.seed(1)
#     random.seed(1)
#
#     meta_test_accuracies = []
#
#     for _ in range(NUM_META_TEST_POINTS):
#         #############################
#         #### YOUR CODE GOES HERE ####
#
#         # sample a batch of test data and partition into
#         # group a (inputa, labela) and group b (inputb, labelb)
#
#         inputa, inputb, labela, labelb = None, None, None, None
#         #############################
#         feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
#
#         result = sess.run([model.total_accuracy1] + model.total_accuracies2, feed_dict)
#         meta_test_accuracies.append(result)
#
#     meta_test_accuracies = np.array(meta_test_accuracies)
#     means = np.mean(meta_test_accuracies, 0)
#     stds = np.std(meta_test_accuracies, 0)
#     ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)
#
#     print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
#     print((means, stds, ci95))
#
#     out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'meta_test_ubs' + str(FLAGS.k_shot) + '_inner_update_lr' + str(FLAGS.inner_update_lr) + '.csv'
#     out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'meta_test_ubs' + str(FLAGS.k_shot) + '_inner_update_lr' + str(FLAGS.inner_update_lr) + '.pkl'
#     with open(out_pkl, 'wb') as f:
#         pickle.dump({'mses': meta_test_accuracies}, f)
#     with open(out_filename, 'w') as f:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerow(['update'+str(i) for i in range(len(means))])
#         writer.writerow(means)
#         writer.writerow(stds)
#         writer.writerow(ci95)
#
def main(args):
    if args.meta_train is False:
        orig_meta_batch_size = args.meta_batch_size
        # always use meta batch size of 1 when testing.
        args.meta_batch_size = 1

    # call data_generator and get data with FLAGS.k_shot*2 samples per class
    data_generator = DataGenerator(args.n_way, args.k_shot*2, args.n_way, args.k_shot*2, config={'data_folder': args.data_path})

    # set up MAML model
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    meta_test_num_inner_updates = args.meta_test_num_inner_updates
    model = MAML(dim_input, dim_output, meta_test_num_inner_updates=meta_test_num_inner_updates)
    model.construct_model(prefix='maml')

    saver = Saver()

    if args.meta_train is False:
        # change to original meta batch size when loading model.
        args.meta_batch_size = orig_meta_batch_size

    if args.meta_train_k_shot == -1:
        args.meta_train_k_shot = args.k_shot
    if args.meta_train_inner_update_lr == -1:
        args.meta_train_inner_update_lr = args.inner_update_lr

    exp_string = 'cls_'+str(args.n_way)+'.mbs_'+str(args.meta_batch_size) + '.k_shot_' + str(args.meta_train_k_shot) + '.inner_numstep' + str(args.num_inner_updates) + '.inner_updatelr' + str(args.meta_train_inner_update_lr)

    resume_itr = 0
    model_file = None

    if args.resume or not args.meta_train:
        model_file = tf.train.latest_checkpoint(args.logdir + '/' + exp_string)
        if args.meta_test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(args.meta_test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.load(model_file)
            # saver.restore(sess, model_file)

    if args.meta_train:
        meta_train(model, saver, exp_string, data_generator, resume_itr)
    else:
        args.meta_batch_size = 1
        meta_test(model, saver, exp_string, data_generator, meta_test_num_inner_updates)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
