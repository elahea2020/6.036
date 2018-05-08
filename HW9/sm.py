
class SM:
    start_state = None  # default start state

    def transition_fn(self, s, i):
        '''s:       the current state
           i:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, i):
        return s + i

    def output_fn(self, s):
        return s
