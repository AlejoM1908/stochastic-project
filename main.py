import argparse
import random


class Node:
    '''
    Used to represent a node in a parsing tree when using CKY algorithm
    '''

    def __init__(self, parent, child1, child2=None, possibility=0.5):
        '''
        Arguments:
            parent: the parent node and a non-terminal symbol from the grammar
            child1: the left child node and a non-terminal symbol from the grammar or a terminal symbol
            child2: the right child node and a non-terminal symbol from the grammar or None when child1 is a terminal symbol
            possibility: probability of the node to appear in the parsing tree
        '''
        self.parent = parent
        self.child1 = child1
        self.child2 = child2
        self.possibility = possibility

    def __repr__(self):
        '''
        :return: the string representation of a Node object.
        '''
        return self.parent

class GrammarGenerator:
    def __init__(self, rules, lexicon):
        '''
        Arguments:
            rules: a list of rules in the grammar
            lexicon: a dictionary of words and their corresponding POS tags
        '''
        self.rules = rules
        self.lexicon = lexicon
        self.readRules(rules)
        self.readLexicon(lexicon)

    def readRules(self, ruleFile):
        '''
        Read the rules from a file and store them in a list.
        :param ruleFile: the name of the file containing the rules
        :return: a list of rules
        '''
        with open(ruleFile, 'r') as f:
            lines = f.readlines()
            self.rules = self.grammar = [x.replace("->", "").split() for x in lines]

    def readLexicon(self, lexiconFile):
        '''
        Read the lexicon from a file and store them in a dictionary.
        :param lexiconFile: the name of the file containing the lexicon
        :return: a dictionary of words and their corresponding POS tags
        '''
        with open(lexiconFile, 'r') as f:
            lines = f.readlines()
            self.lexicon = {x.split(' -> ')[0]: x.split(' -> ')[1].replace('\n','').split(' | ') for x in lines}

    def getPositions(self, terminals):
        '''
        Get a random list of terminal indexes from the list of terminals.
        the list must be at least of lenght 1 and must have non repeating indexes.
        :param terminals: a list of terminals
        :return: a list of indexes
        '''
        positions = []
        while len(positions) == 0:
            positions = random.sample(range(len(terminals)), random.randint(0, len(terminals) - 1))
        return positions

    def getProbabilities(self, length):
        '''
        Given a length, generate a list of length propabilities in range 0 to 1 ramdomly distributed that sum 1.
        :param length: the length of the list
        :return: a list of probabilities
        '''
        probabilities = []
        for i in range(length):
            probabilities.append(random.random())

        probabilities = [round(x / sum(probabilities), 2) for x in probabilities]
        return probabilities


    def generate(self):
        '''
        Generate a grammar from the rules and lexicon.
        
        :return: a matrix of rules for the whole grammar
        '''
        grammar = []
        grammar.extend(self.rules)

        for nonTerminal, terminals in self.lexicon.items():
            positions = self.getPositions(terminals)
            probabilities = self.getProbabilities(len(positions))

            for i in range(len(positions)):
                grammar.append([nonTerminal, f"'{terminals[positions[i]]}'", str(probabilities[i])])

        return grammar

    def writeGrammar(self, grammar, outputFile):
        '''
        Write the grammar to a file.
        :param grammar: the grammar to be written
        :param outputFile: the name of the file to write the grammar to
        :return: None
        '''
        with open(outputFile, 'w') as f:
            for rule in grammar:
                f.write('{} -> {}\n'.format(rule[0], " ".join(rule[1:])))

class Parser:
    '''
    Used to generate the parse tree of a given sentence with a given grammar
    '''

    def __init__(self, grammar, write, draw):
        '''
        Arguments:
            grammar: the grammar file that define the production rules
            sentence: the sentence to parse with the given grammar
            write: a boolean value indicating whether to write the parse trees to a file
            draw: a boolean value indicating whether to draw the parse tree
        '''
        self.parse_table = None
        self.grammar = grammar
        self.write = write
        self.draw = draw

    def parse(self, sentence):
        '''
        Used to parse the sentence with the given grammar using the CKY algorithm.
        '''
        length = len(sentence)
        # self.parse_table[i][j] is the list of nodes in the i+1 cell of j+1 column in the table.
        # we work with the upper-triangular portion of the parse_table
        # In the CKY algorithm, we fill the table a column at a time working from left to right,
        # with each column filled from bottom to top
        self.parse_table = [[[] for i in range(length)] for j in range(length)]

        for j, word in enumerate(sentence):
            # go through every column, from left to right
            for rule in self.grammar:
                # fill the terminal word cell
                if f"'{word}'" == rule[1]:
                    self.parse_table[j][j].append(
                        Node(rule[0], word, possibility=rule[-1]))
            # go through every row, from bottom to top
            for i in range(j - 1, -1, -1):
                for k in range(i, j):
                    child1_cell = self.parse_table[i][k]  # cell left
                    child2_cell = self.parse_table[k + 1][j]  # cell beneath
                    for rule in self.grammar:
                        child1_node = [
                            n for n in child1_cell if n.parent == rule[1]
                        ]
                        if child1_node:
                            child2_node = [
                                n for n in child2_cell if n.parent == rule[2]
                            ]
                            self.parse_table[i][j].extend([
                                Node(rule[0], child1, child2, rule[-1])
                                for child1 in child1_node
                                for child2 in child2_node
                            ])

    def print_tree(self, grammarId):
        '''
        Print and visualize the parse tree starting with the start parent.
        '''
        start_symbol = self.grammar[0][0]
        # final_nodes is the the cell in the upper right hand corner of the parse_table
        # we choose the node whose parent is the start_symbol
        final_nodes = [
            n for n in self.parse_table[0][-1] if n.parent == start_symbol
        ]
        # print(self.parse_table)
        if final_nodes:
            # write all the possible results to a file
            if self.write:
                with open("output.txt", "a") as fw:
                    fw.write(f'Los posible resultados de la gramatica {grammarId} son:\n\n')
                    write_trees = [generate_tree(node) for node in final_nodes]
                    poss_trees = [round(poss_tree(node),4) for node in final_nodes]
                    idx = poss_trees.index(max(poss_trees))
                    for i in range(len(write_trees)):
                        fw.write(f'{i+1})\n{write_trees[i]}\n\nProbabilidad del árbol de parseo: {poss_trees[i]}\n\n')
            # draw the most-likely parse tree
            if self.draw:
                print(
                    "The given sentence is contained in the language produced by the given grammar"
                )
                poss_trees = [poss_tree(node) for node in final_nodes]
                idx = poss_trees.index(max(poss_trees))
                print(generate_tree(final_nodes[idx]))


def generate_tree(node, level=0):
    '''
    Used to generate the string representation of a parse tree.
    '''
    if node.child2 is None:
        return '{}{} -> {}{}'.format('\t' * level, node.parent, '\t' * (level + 1), node.child1).expandtabs(3)
    else:
        return '{}{}\n{}{}\n{}{}'.format(
            '\t' * level, generate_tree(node.child2, level + 1),
            '\t' * level, node.parent, '\t' * level,
            generate_tree(node.child1, level + 1)).expandtabs(3)


def poss_tree(node):
    '''
    :param node: the root node.
    '''
    if node.child2 is None:
        return float(node.possibility)
    return float(node.possibility) * poss_tree(node.child1) * poss_tree(
        node.child2)

def load_sentence(sentence):
        '''
        Used to load the sentence to parse from a file.

        Arguments:
            sentence: the file name containing the sentence to parse.
        '''
        with open(sentence) as fr1:
            input = fr1.readlines()
            result = []

            for line in input:
                result.append(line.split())

            return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rules")
    parser.add_argument("lexicon")
    parser.add_argument("sentence")
    parser.add_argument('--grammars', type=int, default=3)
    parser.add_argument('--no_write', dest='write', action='store_false')
    parser.add_argument('--no_draw', dest='draw', action='store_false')
    parser.set_defaults(write=True, draw=True)
    args = parser.parse_args()

    grammar = GrammarGenerator(args.rules, args.lexicon)
    grammars = []

    for i in range(args.grammars):
        grammars.append(grammar.generate())
        grammar.writeGrammar(grammars[i], f'grammars/grammar-{i}.txt')

    for i in range(args.grammars):
        CKY = Parser(grammars[i], args.write, args.draw)

        sentences = load_sentence(args.sentence)
        for sentence in sentences:
            CKY.parse(sentence)
            CKY.print_tree(i)


if __name__ == '__main__':
    main()
