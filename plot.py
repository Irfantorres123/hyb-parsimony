import matplotlib.pyplot as plt
file="combined1.log"

ga_stats={}
pso_stats={}
hybrid_stats={}

def process_segment(segment:str):
    lines=segment.split("\n")
    #remove empty lines
    lines=[line for line in lines if line.strip()]
    if lines[0].startswith("Testing PSO"):
        stats=pso_stats
    elif lines[0].startswith("Testing Genetic Algorithm"):
        stats=ga_stats
    elif lines[0].startswith("Testing Hybrid Algorithm"):
        stats=hybrid_stats
    else:
        raise ValueError("Unknown segment")
    dataset_name=None
    iteration=False
    accuracy=None
    num_features=None
    for line in lines[1:]:
        if line.startswith("Processing"):
            dataset_name=line.split(":")[-1].strip()
            stats[dataset_name]=[]
            iteration=None
        if dataset_name:
            if line.startswith("Iteration"):
                iteration=True
                accuracy=None
                num_features=None
            if iteration and line.startswith("Best agent accuracy"):
                accuracy=float(line.split(":")[-1].strip())
            if iteration and line.startswith("Best agent num_features"):
                num_features=int(line.split(":")[-1].strip())
                stats[dataset_name].append((accuracy,num_features))
    for dataset in stats.keys():
        data=stats[dataset]
        num_features=[x[1] for x in data]
        iteration=[i for i in range(len(data))]
        plt.plot(iteration,num_features,label=dataset)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Number of features")

    plt.show()

with open(file) as f:
    lines=f.read()
    segments=lines.split("\n\n\n\n\n")
    for segment in segments:
        process_segment(segment)
    iteration=None
    algos=[ga_stats,pso_stats,hybrid_stats]
    algo_names=["Genetic Algorithm","PSO","Hybrid Algorithm"]
    for dataset in pso_stats.keys():
        for i,algo in enumerate(algos):
            data=algo[dataset]
            num_features=[x[1] for x in data]
            iteration=[i for i in range(len(data))]
            plt.plot(iteration,num_features,label=algo_names[i])
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Number of features")
        plt.title(dataset)
        plt.show()
    
        