import pickle


def print_results(results_path='results/very_tiny_cifar10_baseline'):
    print("Showing results for", results_path, ":")

    with open(results_path + '_retrieval.pickle', 'rb') as f:
        results = pickle.load(f)

    # print("Retrieval top-1",  100 * results['cifar10']['raw_precision'][0])
    map_c = 100 * results['cifar10']['map']
    top50_c = 100 * results['cifar10']['raw_precision'][99]

    with open(results_path + '_retrieval_e.pickle', 'rb') as f:
        results = pickle.load(f)
    map_e = 100 * results['cifar10']['map']
    top50_e = 100 * results['cifar10']['raw_precision'][99]

    line = ' $%3.2f$ & $%3.2f$ &  $%3.2f$ & $%3.2f$' % (map_e, map_c, top50_e, top50_c)
    print(line)


if __name__ == '__main__':
    print("Teacher:")
    print_results(results_path='results/resnet18_cifar10_baseline')
    print("Student:")
    print_results(results_path='results/very_tiny_cifar10_baseline')
    print("CNN1-A (Aux.):")
    print_results(results_path='results/aux_pkt')
    print("Proposed:")
    print_results(results_path='results/proposed')
