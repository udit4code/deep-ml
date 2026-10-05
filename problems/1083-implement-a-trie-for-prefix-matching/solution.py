class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False 

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr_node = self.root 
        for ch in word:
            if ch not in curr_node.children:
                curr_node.children[ch] = TrieNode()
            curr_node = curr_node.children[ch]
        curr_node.is_leaf = True 

    def startsWith(self, query: str) -> bool:
        curr_node = self.root 
        for ch in query:
            if ch not in curr_node.children:
                return False
            curr_node = curr_node.children[ch]
        return True

    def search(self, word: str) -> bool:
        curr_node = self.root
        for ch in word:
            if ch not in curr_node.children:
                return False
            curr_node = curr_node.children[ch]
        return curr_node.is_leaf

def trie_operations(words, queries):
    # words: list of strings to insert
    # queries: list of (op, value) tuples where op is 'search' or 'startsWith'
    # return: list of booleans, one per query
    trie = Trie()
    for word in words:
        trie.insert(word)
    result = [ ]
    for (query_op, query_value) in queries:
        if query_op == "search":
            result.append(trie.search(query_value)) 
        elif query_op == "startsWith":
            result.append(trie.startsWith(query_value))
    return result 