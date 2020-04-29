from nn.nn_utils import load_model, save_model
from loaders.cifar_dataset import cifar10_loader
from models.cifar_tiny import Cifar_Tiny
from models.resnet import ResNet18
from nn.retrieval_evaluation import evaluate_model_retrieval
from nn.pkt_transfer import prob_transfer
import torch


def run_transfer(learning_rates=(0.001, 0.0001), iters=(3, 0), method='mds'):
    torch.manual_seed(12345)
    student_layers, teacher_layers, weights, loss_params, T = (3,), (3,), (1,), {}, 2
    print(method)
    transfer_name = method

    # Output paths
    output_path = 'models/aux_' + transfer_name + '.model'
    results_path = 'results/aux_' + transfer_name

    # Load a pre-trained teacher network
    student_net = Cifar_Tiny(10)

    # Use a pre-trained model
    load_model(student_net, 'models/tiny_cifar10.model')

    # Load the teacher model
    teacher_net = ResNet18(num_classes=10)
    load_model(teacher_net, 'models/resnet18_cifar10.model')

    train_loader, test_loader, train_loader_raw = cifar10_loader(batch_size=128)

    # Move the models into GPU
    student_net.cuda()
    teacher_net.cuda()

    # Perform the transfer
    W = None
    for lr, iters in zip(learning_rates, iters):

        if method == 'pkt':
            kernel_parameters = {'student': 'combined', 'teacher': 'combined', 'loss': 'combined'}
            prob_transfer(student_net, teacher_net, train_loader, epochs=iters, lr=lr,
                          teacher_layers=teacher_layers, student_layers=student_layers, layer_weights=weights,
                          kernel_parameters=kernel_parameters, loss_params=loss_params)
        else:
            assert False

    save_model(student_net, output_path)
    print("Model saved at ", output_path)

    # Perform the evaluation
    evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path,
                             result_path=results_path + '_retrieval.pickle', layer=3)
    evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path,
                             result_path=results_path + '_retrieval_e.pickle', layer=3, metric='l2')


if __name__ == '__main__':
    run_transfer(iters=(30, 10), method='pkt')

