import re
from collections import namedtuple

import numpy as np


Transition = namedtuple(typename='Transition', field_names=['source', 'event', 'target'])

Place = namedtuple('Place', ['label', 'marking'])

Arc = namedtuple('Arc', ['source', 'target', 'weight'])

Edge = namedtuple('Edge', ['source', 'target', 'label'])

DiGraph = namedtuple(typename='DiGraph', field_names=['nodes', 'init', 'edges'])


class Automaton(object):

    def __init__(self, states, init, events, trans, marked=None, forbidden=None):
        """
        This is the constructor of the automaton.

        At creation, the automaton gets the following attributes assigned:
        :param states: A set of states
        :param init: The initial state
        :param events: A set of events
        :param trans: A set of transitions
        :param marked: (Optional) A set of marked states
        :param forbidden: (Optional) A set of forbidden states
        """
        self.states = states
        self.init = init
        self.events = events
        self.trans = trans
        self.marked = marked if marked else set()
        self.forbidden = forbidden if forbidden else set()

    def __str__(self):
        """Prints the automaton in a pretty way."""
        return 'states: \n\t{}\n' \
               'init: \n\t{}\n' \
               'events: \n\t{}\n' \
               'transitions: \n\t{}\n' \
               'marked: \n\t{}\n' \
               'forbidden: \n\t{}\n'.format(
                   self.states, self.init, self.events,
                   '\n\t'.join([str(t) for t in self.trans]), self.marked, self.forbidden)

    def __setattr__(self, name, value):
        """Validates and protects the attributes of the automaton"""
        if name in ('states', 'events'):
            value = frozenset(self._validate_set(value))
        elif name == 'init':
            value = self._validate_init(value)
        elif name == 'trans':
            value = frozenset(self._validate_transitions(value))
        elif name in ('marked', 'forbidden'):
            value = frozenset(self._validate_subset(value))
        super(Automaton, self).__setattr__(name, value)

    def __getattribute__(self, name):
        """Returns a regular set of the accessed attribute"""
        if name in ('states', 'events', 'trans', 'marked', 'forbidden'):
            return set(super(Automaton, self).__getattribute__(name))
        else:
            return super(Automaton, self).__getattribute__(name)

    def __eq__(self, other):
        """Checks if two Automata are the same"""
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def _validate_set(states):
        """Checks that states is a set and the states in it are strings or integers"""
        assert isinstance(states, set)
        for state in states:
            assert isinstance(state, str) or isinstance(
                state, int), 'A state must be either of type string or integer!'
        return states

    def _validate_subset(self, subset):
        """Validates the set and checks whether the states in the subset are part of the state set"""
        subset = self._validate_set(subset)
        assert subset.issubset(
            self.states), 'Marked and forbidden states must be subsets of all states!'
        return subset

    def _validate_init(self, state):
        """Checks whether the state is part of the state set"""
        assert isinstance(state, str) or isinstance(
            state, int), 'The initial state must be of type string or integer!'
        assert state in self.states, 'The initial state must be member of states!'
        return state

    def _validate_transitions(self, transitions):
        """Checks that all transition elements are part in the respective sets (states, events)"""
        assert isinstance(transitions, set)
        for transition in transitions:
            assert isinstance(transition, Transition)
            assert transition.source in self.states
            assert transition.event in self.events
            assert transition.target in self.states
        return transitions


class PetriNet(object):

    def __init__(self, places, transitions, arcs):
        """
        This is the constructor of the PetriNet.

        At creation, the PetriNet gets the following attributes assigned:
        :param places: A set of Places
        :param transitions: A set of Transitions
        :param arcs: A set of Arcs
        """
        assert isinstance(places, list)
        self.places = places
        assert isinstance(arcs, set)
        self.arcs = arcs
        assert isinstance(transitions, set)
        self.transitions = transitions
        
        self.P = np.array([p.label for p in places], str)
        self.init_marking = np.array([p.marking for p in places], int)
        self.T = np.array(list(transitions))
        self.A_minus = np.zeros((len(places), len(transitions)), int)
        self.A_plus = np.zeros((len(places), len(transitions)), int)
        # populate transition matrices
        for a in arcs:
            if a.source in self.P:
                n, = np.where(self.P == a.source)
                m, = np.where(self.T == a.target)
                self.A_minus[n, m] = a.weight
            else:
                n, = np.where(self.P == a.target)
                m, = np.where(self.T == a.source)
                self.A_plus[n, m] = a.weight

    def make_reachability_graph(self):
        """Computes the reachability graph of the PetriNet"""        
        new_markings = [self.init_marking]
        nodes = {array_str(self.init_marking)}
        edges = set()

        while new_markings:
            m = new_markings.pop()
            m = np.reshape(m, [-1, 1])
            for i, t in enumerate(self.T):
                s = np.zeros([len(self.T), 1], int)
                s[i] = 1
                if np.all(m >= np.matmul(self.A_minus, s)):
                    m_plus = m + np.matmul(self.A_plus - self.A_minus, s)
                    m_plus_str = array_str(m_plus)
                    edges.add(Edge(array_str(m), m_plus_str, t))
                    if m_plus_str not in nodes:
                        nodes.add(m_plus_str)
                        new_markings.append(m_plus)
        return DiGraph(nodes, array_str(self.init_marking), edges)
    

def array_str(a):
    """Casts numpy array to string and removes superfluous whitespaces"""
    return re.sub(" +", " ", str(a.flatten())).replace('[ ', '[')