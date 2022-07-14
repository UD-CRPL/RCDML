import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

#sample_df =
#inhib_df =

def all_in_one(result_path, data, ids, value):
    #np.random.seed(667)
    number_of_plots = len(ids)
    print(number_of_plots)
    gridsize = get_subplots_gridsize(number_of_plots)
    print(gridsize)
    fig, axis = plt.subplots(gridsize[0], gridsize[1], sharex=True, sharey=True)
    fig.suptitle('Drug Response AUC Distribution for 24 RTK-TYPE-III Inhibitors')

    #colors = []
    #for i in range(0, len(combinations)):
    #    color=np.random.rand(3,)
    #    colors.append(color)

    k = 0
    for i in range(0, gridsize[0]):
        for j in range(0, gridsize[1]):
            #axis[i, j].plot([0, 1], [0, 1], color='darkblue', linestyle='--')
            #for iter, combo in enumerate(combinations):
                #auc_score = auc(data[k][iter][0], data[k][iter][1])
            sns.histplot(dataset[dataset['inhibitor'] == list(ids)[k]][value], kde=True, stat = 'count', ax=axis[i, j])
            ax2 = axis[i,j].twinx()
            sns.boxplot(x=value, data = dataset[dataset['inhibitor'] == list(ids)[k]], palette="Set2", ax=ax2, boxprops=dict(alpha=.275))
            #sns.boxplot(x=value, data = dataset[dataset['inhibitor'] == list(ids)[k]], ax=axis[i, j], boxprops=dict(alpha=.3))
            axis[i, j].set(xlabel='', ylabel='')
                #axis[i, j].plot(data[k][iter][0], data[k][iter][1], color=colors[iter], label= combo[0].upper() + " + " + combo[1].upper(), alpha = .6)
            title = "Samples: " + str(dataset[dataset['inhibitor'] == list(ids)[k]]["counts"].iloc[0]) + "; Inhibitor: " + list(ids)[k].split(' ')[0].upper()
            axis[i, j].set_title(title)
            k = k + 1
    #for ax in axis.flat:
        #ax.set(xlabel='1 - Specificity (False Positive Rate)', ylabel='Sensitivity (True Positive Rate)')
    import string
    abc = list(string.ascii_uppercase)

    for label, ax in enumerate(axis.flat):
        #ax.set_title('Normal Title', fontstyle='italic')
        #ax.set_title(abc[label] + ")", fontfamily='serif', loc='left', fontsize='medium')
        ax.set_title(str(label + 1) + ")", fontfamily='serif', loc='left', fontsize='medium')

    for ax in axis.flat:
    #    ax.set(xlabel='1 - Specificity (False Positive Rate)', ylabel='Sensitivity (True Positive Rate)')
        ax.label_outer()

    fig.text(0.5, 0.04, 'AUC', ha='center')
    fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')
    fig.set_figheight(15)
    fig.set_figwidth(22)
    #plt.figure(figsize=(50, 50))
    #plt.legend(framealpha = .3, loc=(1.04, 0))
    #plt.show()
    plt.savefig(result_path + "/all_distributions.png")
    return

#sns.set_theme(style="whitegrid")
def plot_normal_distribution(url, ax_box, ax_hist, dataset, title, value):
    sns.set(style="ticks")
#x = np.random.randn(100)
    sns.boxplot(x=value, data = dataset, ax=ax_box).set_title(title)
    sns.histplot(dataset[value], kde=True, stat = 'count', ax=ax_hist)

    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    #ax = sns.boxplot(x="inhibitor", y="ic50", data=dataset, palette="Set3")
    #ax = sns.boxplot(x="lab_id", y="ic50", data=dataset, palette="Set3")
    plt.savefig(url)
    #plt.cla()
    ax_box.clear()
    ax_hist.clear()
    #plt.show()

#for sample in samples:
#    plot_distribution()

def distribution_plot(level, ids, dataset, value):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                    gridspec_kw={"height_ratios": (.15, .85)})
    for i, id in enumerate(ids):
        if level == 'inhibitor':
            print("Plotting Inhibitor: " + id + " (" + str(i) + "/" + str(len(ids)) + ")")
            inhib_df = dataset[dataset['inhibitor'] == id]
        #print(inhib_df)
        # for sanity check, make sure rows (samples) in df match with count value
            count = inhib_df['counts'].iloc[0]
            count == inhib_df.shape[0]
            title = value.upper() + " Distribution Across " + str(count) + " Samples for Inhibitor: " + id
            plot_path = result_path + "Inhibitor/" + value + "/plots/"
            id_path = result_path + 'Inhibitor/' + value + '/sample_ids/'
            make_result_dir(plot_path)
            make_result_dir(id_path)
            inhib_df.to_csv(id_path + id + value + " .tsv", sep ='\t', index = False)
            plot_normal_distribution(plot_path + id + " " + value.upper() + " Distribution.png", ax_box, ax_hist, inhib_df, title, value)
        elif level == 'samples':
            print("Plotting Sample: " + id + " (" + str(i) + "/" + str(len(ids)) + ")")
            sample_df = dataset[dataset['lab_id'] == id]
            #print(sample_df)
            #print(inhib_df)
            # for sanity check, make sure rows (samples) in df match with count value
            count = sample_df.shape[0]
            title = value.upper() + " Distribution Across " + str(count) + " Inhibitors for Sample: " + id
            plot_path = result_path + "Samples/" + value + "/plots/"
            id_path = result_path + 'Samples/' + value + '/inhibitor_ids/'
            make_result_dir(plot_path)
            make_result_dir(id_path)
            sample_df.to_csv(id_path + id + "_" + value + ".tsv", sep ='\t', index = False)
            plot_normal_distribution(plot_path + id + " " + value.upper() + " Distribution.png", ax_box, ax_hist, sample_df, title, value)
        else:
            sys.exit("Invalid level chosen for plot distribution")

#inhibitor_plot(inhibitor)
#sample_plot()
def get_statistics(dataset):
    inhibitor_df = dataset.groupby('inhibitor').agg({'auc':['describe', 'median']})
    sample_df = dataset.groupby('lab_id').agg({'auc':['describe', 'median']})

    inhibitor_df.columns = inhibitor_df.columns.droplevel().droplevel()
    sample_df.columns = sample_df.columns.droplevel().droplevel()
    inhibitor_df = inhibitor_df.reset_index().rename(columns={"auc": "median"})
    sample_df = sample_df.reset_index().rename(columns={"auc": "median"})

    correct_order = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']

    #print(inhibitor_df)
    inhibitor_df = inhibitor_df.reindex(columns = correct_order.insert(0, 'inhibitor'))
    sample_df = sample_df.reindex(columns = correct_order.insert(0, 'lab_id'))

    #inhibitor_df.to_csv(result_path + 'Inhibitor/auc_distribution_statistics.tsv', sep ='\t', index = False)
    #sample_df.to_csv(result_path + 'Samples/auc_distribution_statistics.tsv', sep ='\t', index = False)
    return inhibitor_df, sample_df
### BUILD DATA MATRICES
#inhibitor_plot()
#sample_plot()
#    plot_distribution()

def drug_response_plot(url, dataset, inhibitor, samples):
    #inhibitor_means = inhibitor[['inhibitor', 'mean']].set_index('inhibitor').to_dict()
    inhibitor_means = inhibitor.set_index('inhibitor')['mean'].to_dict()
    sample_means = samples.set_index('lab_id')['mean'].to_dict()
    #print(dataset)
    dataset[['inhib_distance', 'inhib_mean']] = dataset.apply(lambda x: [((x['auc'] - inhibitor_means[x['inhibitor']]) ** 2) * (x['auc']/inhibitor_means[x['inhibitor']]), inhibitor_means[x['inhibitor']]], axis = 1, result_type ='expand')
    dataset[['sample_distance', 'sample_mean']] = dataset.apply(lambda x: [(x['auc'] - sample_means[x['lab_id']]) ** 2, sample_means[x['lab_id']]], axis = 1, result_type ='expand')
    #print(dataset)
    #dataset = dataset.drop('ic50', axis = 1)
    get_plot(url, dataset.groupby('inhibitor'))

from matplotlib.gridspec import GridSpec


def get_plot(url, dataset):
    number = 0
    #fig = plt.figure()
    #gs = GridSpec(4, 4)

    #ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    #ax_hist_y = fig.add_subplot(gs[0, 0:3])
    #ax_hist_x = fig.add_subplot(gs[1:4, 3])
    sns.set(style="ticks")
    for i, g in dataset:
        #if number > 2:
        #    return
        print("Inhibitor: " + i + " (" + str(number + 1) +"/" + str(len(dataset)) + ")")
        #print(str(number))
        p = sns.jointplot(x="sample_distance",
                      y="inhib_distance",
                     data=g)
        #plt.tight_layout()
        #g = g.sort_values('inhib_distance')
        #ax_scatter.scatter(g['sample_distance'], g['inhib_distance'], label = g['inhibitor'].iloc[0], marker='.', alpha=0.5)
        #ax_hist_y.hist(g['inhib_distance'], color='tab:blue', alpha=0.4)
        #ax_hist_x.hist(g['sample_distance'], orientation = 'horizontal', color='tab:blue', alpha=0.4)
        number = number + 1
    # Sets graph axis labels, title, legend table
        #ax_scatter.set_xlabel('Sample Distance')
        #ax_scatter.set_ylabel('Inhibitor Distance')
        #ax_scatter.set_title(g['inhibitor'].iloc[0])
        p.fig.suptitle(g['inhibitor'].iloc[0])

        p.ax_joint.axvline(x=g.sample_distance.median(),linestyle='--', c = 'red')
        p.ax_joint.axhline(y=g.inhib_distance.median(),linestyle='--', c = 'red')
        #ax_scatter.legend()
        # Saves the graph as a png file
        #plt.show()
        plt.savefig(url + g['inhibitor'].iloc[0] + '.png')
        g.to_csv(url + g['inhibitor'].iloc[0] + '.csv', sep ='\t', index = False)
        plt.clf()
        #plt.savefig(ur)
        #ax_scatter.clear()
        #ax_hist_y.clear()
        #ax_hist_x.clear()
    return

def make_result_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def make_labels(dataset, result_path):
    #dataset['lab_id'] =
    dataset = dataset[dataset['counts'] > 300]
    dataset = dataset.drop('counts', axis = 1)
    for i, inhib in enumerate(dataset['inhibitor'].drop_duplicates()):
        labels = dataset[dataset['inhibitor'] == inhib]
        labels = labels[['lab_id', 'auc']]
        labels['lab_id'] = labels['lab_id'].str.replace("-","X", regex=True)
        labels['lab_id'] = 'X' + labels['lab_id'].astype(str)
        q1 = labels['auc'].quantile(.25)
        q3 = labels['auc'].quantile(.75)
        labels['GROUP'] = labels['auc'].apply(lambda x: auc_to_binary(x, q1, q3))
        labels = labels[labels['GROUP'].isin([0, 1])]
        labels = labels.drop('auc', axis = 1)
        labels = pd.get_dummies(labels, columns = ['GROUP'])
        labels = labels.rename(columns = {'lab_id':'Sample', 'GROUP_0':'low', 'GROUP_1':'high'})
        print(inhib)
        labels.to_csv(result_path + inhib + ".txt", sep = '\t', index = False)

def auc_to_binary(value, q1, q3):
    if value >= q3:
        return 1
    elif value <= q1:
        return 0
    else:
        return -1

def get_drug_names(dataset):
    drug_list = pd.read_excel(path + "aml/beatAML/variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
    drug_list = drug_list[drug_list["family"] == drug_family]

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


def find_middle(input_list):
    middle = float(len(input_list)) / 2
    if len(input_list) % 2 != 0:
        return (input_list[int(middle - .5)], input_list[int(middle - .5)])
    else:
        return (input_list[int(middle)], input_list[int(middle - 1)])

def get_subplots_gridsize(num):
    factors = factorize(num)
    middle = find_middle(factors)
    return middle

# DARWIN MACRO
dataset_path = "/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/"
result_path = '/Users/mf0082/Documents/Nature_Comm_paper/results/drug_distributions/'
drug_family = "RTK_TYPE_III"
## LOADING DATASET
dataset = pd.read_excel(dataset_path + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses")
family = pd.read_excel(dataset_path + "variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
family = family[family["family"] == drug_family]
dataset = dataset[dataset['inhibitor'].isin(family['inhibitor'])]

samples = dataset['lab_id.1']
samples = samples.drop_duplicates().dropna()

# DEBUG
inhibitor_count = dataset[['inhibitor.1', 'Sample counts']].dropna()
inhibitor_count = inhibitor_count[inhibitor_count['Sample counts'] > 300]
inhibitor = inhibitor_count['inhibitor.1']

dataset = dataset[['inhibitor', 'lab_id', 'ic50', 'auc', 'counts']]
dataset = dataset[dataset['counts'] > 300]


samples = pd.concat((samples, dataset['lab_id']), axis = 0)
samples = samples.drop_duplicates()
inhibitor = pd.concat((inhibitor, dataset['inhibitor']), axis = 0)
inhibitor = inhibitor.drop_duplicates()

make_result_dir(result_path)

print("FAMILY")
print(family)
print("DATASET")
print(dataset)
print("INHIBITOR")
print(inhibitor)

#make_labels(dataset, result_path)
#make_result_dir(result_path + "/distance_plots/")

all_in_one(result_path, dataset, inhibitor, "auc")
#distribution_plot('samples', samples, dataset, 'auc')
#distribution_plot('inhibitor', inhibitor, dataset, 'auc')
#distribution_plot('samples', samples, dataset, 'ic50')
#distribution_plot('inhibitor', inhibitor, dataset, 'ic50')
#inhibitor_df, sample_df = get_statistics(dataset)
#drug_response_plot(result_path + "/distance_plots/", dataset, inhibitor_df, sample_df)
