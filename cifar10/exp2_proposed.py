from nn.nn_utils import load_model, save_model
from loaders.cifar_dataset import cifar10_loader
from models.cifar_tiny import Cifar_Tiny
from models.cifar_very_tiny import Cifar_Very_Tiny
from nn.retrieval_evaluation import evaluate_model_retrieval
from nn.pkt_transfer import prob_transfer
import torch
import numpy as np

def run_transfer(learning_rates=(0.001, ), epochs=(10,), decay=0.7, init_weight=100):
    torch.manual_seed(12345)
    print(init_weight, decay)
    student_layers, teacher_layers, loss_params, T = (3, 2, 1, 0), (3, 2, 1, 0), {}, 2

    # Output paths
    output_path = 'models/proposed.model'
    results_path = 'results/proposed'

    # Load a pre-trained teacher network
    student_net = Cifar_Very_Tiny(10)

    # Load the teacher model
    teacher_net = Cifar_Tiny(num_classes=10)
    load_model(teacher_net, 'models/aux_pkt.model')
    
    kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
    train_loader, test_loader, train_loader_raw = cifar10_loader(batch_size=128)
    
    # Move the models into GPU
    student_net.cuda()
    teacher_net.cuda()
    
    np.random.seed(1)
    
    cur_weight = init_weight
    for cur_epoch, cur_lr in zip(epochs, learning_rates):
        print("Running for ", cur_epoch, " epochs with lr = ", cur_lr)
        for i in range(cur_epoch):
            print(cur_weight)
            weights = (1, cur_weight, cur_weight, cur_weight)
            prob_transfer(student_net, teacher_net, train_loader, epochs=1, lr=cur_lr,
                          teacher_layers=teacher_layers, student_layers=student_layers, layer_weights=weights,
                          kernel_parameters=kernel_parameters, loss_params=loss_params)
            cur_weight = cur_weight * decay


    save_model(student_net, output_path)
    print("Model saved at ", output_path)

    # Perform the evaluation

    evaluate_model_retrieval(net=student_net, path='',
                             result_path=results_path + '_retrieval.pickle', layer=3)
    evaluate_model_retrieval(net=student_net, path='',
                             result_path=results_path + '_retrieval_e.pickle', layer=3, metric='l2')


if __name__ == '__main__':

    run_transfer(learning_rates=(0.001, ), epochs=(50,))
