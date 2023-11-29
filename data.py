import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

FIG_SIZE = 5


def load_UNINet(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    lines = data.split('\n')

    N, NM = [int(value.split('=')[1]) for value in lines[1].split()]

    if "ROW LABELS:" in lines:
        labels_index = lines.index("ROW LABELS:")
        node_names = lines[labels_index + 1:labels_index + 1 + N]
    else:
        node_names = None
        
    # get level labels
    layer_labels_index = lines.index("LEVEL LABELS:")
    layer_labels = lines[layer_labels_index + 1:layer_labels_index + 1 + NM]
    
    start_index = lines.index("DATA:")
    MLG = []
    for i in range(NM):
        matrix_lines = lines[start_index + 1 + i * N:start_index + 1 + (i + 1) * N]
        matrix = np.array([list(map(int, line.split())) for line in matrix_lines])
        graph = nx.from_numpy_array(matrix)
        MLG.append(graph)

    return MLG, layer_labels, node_names


def display_MLG(MLG, layer_labels):

    M = len(MLG)
    
    fig, axes = plt.subplots(1, M, figsize=(FIG_SIZE * M, FIG_SIZE))

    for i, G in enumerate(MLG):
        nx.draw(G, with_labels=False, node_size=50, node_color='black', edge_color='black', ax=axes[i])
        axes[i].set_title(f"{layer_labels[i]}")

    plt.show()

    adjacency_matrices = [nx.adjacency_matrix(G) for G in MLG]
    fig, axes = plt.subplots(1, M, figsize=(FIG_SIZE * M, FIG_SIZE))
    for i, adjacency_matrix in enumerate(adjacency_matrices):
        axes[i].spy(adjacency_matrix, markersize=2, marker="D", color="Blue")
        axes[i].set_title(f"{layer_labels[i]}")
    plt.show()

# AUCS Dataset #####################################################
def init_graph():
    path = './datasets/AUCS/aucs_nodelist.txt'
    g = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            if line[1] == 'NA':
                continue
            else:
                g.add_node(line[0])
    return g


def get_true_labels():
    true_labels = []
    na_list = []
    path = './datasets/AUCS/aucs_nodelist.txt'
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            t = line[1]
            if t == 'NA':
                na_list.append(line[0])
            else:
                true_labels.append(int(t[-1])-1)
    return true_labels, na_list


def load_AUCS():
    path = './datasets/AUCS/aucs_edgelist.txt'

    # Declare each layer's graph
    lunch = init_graph()
    facebook = init_graph()
    leisure = init_graph()
    work = init_graph()
    coauthor = init_graph()
    table = {
        'lunch': lunch,
        'facebook': facebook,
        'leisure': leisure,
        'work': work,
        'coauthor': coauthor,
    }
    true_labels, na = get_true_labels()

    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            name = line[2]
            if line[0] in na or line[1] in na:
                continue
            else:
                table[name].add_edge(line[0], line[1])

    MLG = [lunch, facebook, leisure, work, coauthor]

    layer_labels = list(table.keys())

    return MLG, layer_labels, true_labels

# MIT Dataset #####################################################

    

def extract_friendship_graph():
    df = pd.read_csv('datasets/MIT/reality-mining-survey.txt', sep='\t')
    # start with 94 nodes from 0 to 89
    g = nx.Graph()
    g.add_nodes_from(range(90))
    g.add_weighted_edges_from(df[['id1', 'id2', 'close-friends?']].values)
    return g

def extract_proximity_graph():
    df = pd.read_csv('datasets/MIT/reality-mining-proximity.txt', sep='\t')
    df['start'] = pd.to_datetime(df['start'], format='mixed')
    df['end'] = pd.to_datetime(df['end'], format='mixed')

    df['duration'] = df['end'] - df['start']
    df['duration'] = df['duration'].dt.total_seconds()

    #  add a column to measure how many 30 mn intervals are in the duration of each proximity event
    df['duration_30mn'] = df['duration'] / 1800 + 1
    df['duration_30mn'] = df['duration_30mn'].astype(int)

    df = df[['id1', 'id2', 'duration_30mn']].groupby(['id1', 'id2']).sum().reset_index()
    df = df.rename(columns={'duration_30mn': 'proximity'})

    g = nx.Graph()
    g.add_nodes_from(range(90))

    g.add_weighted_edges_from(df[['id1', 'id2', 'proximity']].values)
    return g

def extract_calls_graph():
    df = pd.read_csv('datasets/MIT/reality-mining-calls.txt', sep='\t')
    df = df[df.subjectId != df.otherPartyId]
    incoming_mask = df['direction'] == 'Incoming'
    df.loc[incoming_mask, ['subjectId', 'otherPartyId']] = df.loc[incoming_mask, ['otherPartyId', 'subjectId']].values
    df.loc[incoming_mask, 'direction'] = 'Outgoing'
    df = df.drop_duplicates(subset=['subjectId', 'otherPartyId', 'direction', 'duration'])
    df = df.groupby(['subjectId', 'otherPartyId']).count().reset_index()
    df = df.rename(columns={'duration': 'calls'})

    df['id_pair'] = df.apply(lambda row: tuple(sorted([row['subjectId'], row['otherPartyId']])), axis=1)
    incoming_mask = df['direction'] == 'incoming'

    df.loc[incoming_mask, ['subjectId', 'otherPartyId']] = df.loc[incoming_mask, ['otherPartyId', 'subjectId']].values

    df.sort_values(by=['id_pair', 'direction'], inplace=True)
    df.drop_duplicates(subset=['id_pair', 'direction'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.groupby(['id_pair']).sum().reset_index()

    df['subjectId'] = df['id_pair'].apply(lambda x: x[0])
    df['otherPartyId'] = df['id_pair'].apply(lambda x: x[1])

    g = nx.Graph()
    g.add_nodes_from(range(90))
    g.add_weighted_edges_from(df[['subjectId', 'otherPartyId', 'calls']].values)
    return g

def extract_affiliation():
    affiliation = pd.read_csv('datasets/MIT/reality-mining-labels.txt', header=None)[0].tolist()
    return affiliation

def preprocess_MIT():
    friendship = extract_friendship_graph()
    friendship_adj = pd.DataFrame(nx.adjacency_matrix(friendship).todense(), dtype=int)
    friendship_adj.to_csv("datasets/MIT/friendship.txt", header=None, index=None, sep=' ')

    calls = extract_calls_graph()
    calls_adj = pd.DataFrame(nx.adjacency_matrix(calls).todense(), dtype=int)
    calls_adj.to_csv("datasets/MIT/calls.txt", header=None, index=None, sep=' ')

    proximity = extract_proximity_graph()
    proximity_adj = pd.DataFrame(nx.adjacency_matrix(proximity).todense(), dtype=int)
    proximity_adj.to_csv("datasets/MIT/proximity.txt", header=None, index=None, sep=' ')

    affiliation = extract_affiliation()
    affiliation = pd.DataFrame(affiliation)
    affiliation.to_csv("datasets/MIT/affiliation.txt", header=None, index=None, sep=' ')


def load_MIT(preprocess=False):
    if preprocess:
        preprocess_MIT()
    MLG = []
    layer_labels = ['friendship', 'calls', 'proximity']
    for layer in layer_labels:
        adj = pd.read_csv(f'datasets/MIT/{layer}.txt', header=None, sep=' ')
        g = nx.from_numpy_array(adj.values)
        MLG.append(g)

    true_labels = pd.read_csv('datasets/MIT/affiliation.txt', header=None, sep=' ')[0].tolist()

    return MLG, layer_labels, true_labels


# Cora Dataset #####################################################



def extract_informative_words(s):
    import re
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')

    s = s.lower()
    s = re.sub(r'[^a-z\s]', '', s)

    words = s.split()

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return words

def extract_name_and_label(s):
    name = s.split("/")[-1]
    if len(s.split("/")) > 1:
        label = s.split("/")[-2]
    else:
        label = None
    return name, label

def extract_paper_info():
    with open("datasets/Cora/citations.withauthors") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    paper_start_indices = [i for i, line in enumerate(lines) if line == "***"]
    authors_start_indices = [i for i, line in enumerate(lines) if line == "*"]

    papers = {}
    for i, start in enumerate(paper_start_indices[:-1]):
        paper_id = int(lines[start+1])
        name, label = extract_name_and_label(lines[start+2])
        papers[paper_id] = {
            "name" : name,
            "label" : label,
            "cited" : [int(p) for p in lines[start+3:authors_start_indices[i]]],
            "authors" : lines[authors_start_indices[i]+1:paper_start_indices[i+1]]
        }

    with open("datasets/Cora/papers") as f:
        for line in f:
            if len(line.strip().split("\t")) <= 2:
                continue
            paper_id, name, desc = line.strip().split("\t")
            paper_id = int(paper_id)
            if paper_id in papers.keys():
                desc = line.strip().split("\t")[2]
                title = desc.split("<title>")[1].split("</title>")[0]
                title = extract_informative_words(title)
                papers[paper_id]["title"] = title
        return papers

def preprocess_authors(papers):
    from sklearn.metrics.pairwise import cosine_similarity
    authors = [paper["authors"] for paper in papers.values()]
    author_to_index = {author:i for i, author_list in enumerate(authors) for author in author_list}
    num_authors = len(author_to_index)
    author_vectors = np.zeros((len(papers), num_authors))
    for i, author_list in enumerate(authors):
        for author in author_list:
            author_vectors[i, author_to_index[author]] = 1
    cosine_sim = cosine_similarity(author_vectors)
    np.fill_diagonal(cosine_sim, 0)
    pd.DataFrame(cosine_sim).to_csv("datasets/Cora/authors.txt", header=None, index=None, sep=' ')
    


def preprocess_titles(papers, paper_index_to_id):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = CountVectorizer()
    titles = [' '.join(papers[paper_index_to_id[i]]["title"]) for i in range(len(papers))]
    vectorized_titles = vectorizer.fit_transform(titles)
    cosine_sim = cosine_similarity(vectorized_titles)
    np.fill_diagonal(cosine_sim, 0)
    pd.DataFrame(cosine_sim).to_csv("datasets/Cora/titles.txt", header=None, index=None, sep=' ')    

def preprocess_citations(papers, paper_index_to_id):
    citation = nx.Graph()
    for i in range(len(papers)):
        for paper_id in papers[paper_index_to_id[i]]["cited"]:
            if paper_id in paper_index_to_id.keys():
                citation.add_edge(i, paper_index_to_id[paper_id])
    adjacency_matrix = nx.adjacency_matrix(citation)
    pd.DataFrame(adjacency_matrix.todense()).to_csv("datasets/Cora/citations.txt", header=None, index=None, sep=' ')
    
def preprocess_labels(papers, paper_index_to_id):
    labels = [papers[paper_index_to_id[i]]["label"] for i in range(len(papers))]
    pd.DataFrame(labels).to_csv("datasets/Cora/labels.txt", header=None, index=None, sep=' ')

def preprocess_Cora(classes=["Robotics", "NLP", "Data_Mining"]):
    papers = extract_paper_info()
    papers = {k:v for k,v in papers.items() if "title" in v.keys() and v["label"] in ["Robotics", "NLP", "Data_Mining"]} 
    paper_index_to_id = {i:k for i, k in enumerate(papers.keys())}

    preprocess_authors(papers)
    preprocess_titles(papers, paper_index_to_id)
    preprocess_citations(papers, paper_index_to_id)
    preprocess_labels(papers, paper_index_to_id)


def load_Cora(preprocess=False):
    if preprocess:
        preprocess_Cora()
    MLG = []
    layer_labels = ['authors', 'titles', 'citations']
    for layer in layer_labels:
        adj = pd.read_csv(f'datasets/Cora/{layer}.txt', header=None, sep=' ')
        g = nx.from_numpy_array(adj.values)
        MLG.append(g)

    true_labels = pd.read_csv('datasets/Cora/labels.txt', header=None, sep=' ')[0].tolist()

    return MLG, layer_labels, true_labels


# Dataloader #####################################################

class Dataset():
    def __init__(self, name, **kwargs):
        if name == "AUCS":
            MLG, layer_labels, labels = load_AUCS()
        elif name == "MIT":
            MLG, layer_labels, labels = load_MIT(**kwargs)
        elif name == "Cora":
            MLG, layer_labels, labels = load_Cora(**kwargs)
        elif name == "UNINet":
            MLG, layer_labels, labels = load_UNINet(**kwargs)
        else:
            raise ValueError(f"Dataset {name} not found")
        
        self.MLG = MLG
        self.layer_labels = layer_labels
        self.labels = labels
        
    def display(self):
        display_MLG(self.MLG, self.layer_labels)
