from sklearn.datasets import fetch_20newsgroups

def load_data():
    data = fetch_20newsgroups(subset="all")
    return data.data

if __name__ == "__main__":
    docs = load_data()
    print(len(docs))
