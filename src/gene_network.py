"""
Network structure of myeloid differentiation. 
Ref: Hierarchical Differentiation of Myeloid Progenitors Is Encoded in the Transcription Factor Network, PLos One
"""

from enum import IntEnum
import numpy as np


class Genes(IntEnum):
    """
    Define the genes in this network.
    """
    GATA2 = 0
    GATA1 = 1
    FOG1 = 2
    EKLF = 3
    Fli1 = 4
    SCL = 5
    CEBPa = 6
    PU1 = 7
    cJun = 8
    EgrNab = 9
    Gfi1 = 10

# The true regulator list for each gene
Regulators = {Genes.GATA2: {Genes.GATA2, Genes.GATA1, Genes.FOG1, Genes.PU1},
              Genes.GATA1: {Genes.GATA2, Genes.GATA1, Genes.Fli1, Genes.PU1},
              Genes.FOG1: {Genes.GATA1},
              Genes.EKLF: {Genes.GATA1, Genes.Fli1},
              Genes.Fli1: {Genes.GATA1, Genes.EKLF},
              Genes.SCL: {Genes.GATA1, Genes.PU1},
              Genes.CEBPa: {Genes.GATA1, Genes.FOG1, Genes.SCL, Genes.CEBPa},
              Genes.PU1: {Genes.GATA2, Genes.GATA1, Genes.CEBPa, Genes.PU1},
              Genes.cJun: {Genes.PU1, Genes.Gfi1},
              Genes.EgrNab: {Genes.PU1, Genes.cJun, Genes.Gfi1},
              Genes.Gfi1: {Genes.CEBPa, Genes.EgrNab}}

# Attractors of this network (actually all fixed points)
Fixed_points = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])


def is_fixed_point(s):
    """
    Whether a given state is the fixed point of this network
    :param s: a state
    :return: True if it is, else False
    """
    for fixed_point in Fixed_points:
        if np.array_equal(s, fixed_point):
            return True
    return False


def _update_GATA2(s):
    """
    update GATA-2
    :param s: the old state
    :return: the new state by updating GATA-2
    """
    new_s = np.copy(s)
    new_s[Genes.GATA2] = s[Genes.GATA2] and not (s[Genes.GATA1] and s[Genes.FOG1]) and not s[Genes.PU1]
    return new_s


def _update_GATA1(s):
    new_s = np.copy(s)
    new_s[Genes.GATA1] = (s[Genes.GATA1] or s[Genes.GATA2] or s[Genes.Fli1]) and not s[Genes.PU1]
    return new_s


def _update_FOG1(s):
    new_s = np.copy(s)
    new_s[Genes.FOG1] = s[Genes.GATA1]
    return new_s


def _update_EKLF(s):
    new_s = np.copy(s)
    new_s[Genes.EKLF] = s[Genes.GATA1] and not s[Genes.Fli1]
    return new_s


def _update_Fli1(s):
    new_s = np.copy(s)
    new_s[Genes.Fli1] = s[Genes.GATA1] and not s[Genes.EKLF]
    return new_s


def _update_SCL(s):
    new_s = np.copy(s)
    new_s[Genes.SCL] = s[Genes.GATA1] and not s[Genes.PU1]
    return new_s


def _update_CEBPa(s):
    new_s = np.copy(s)
    new_s[Genes.CEBPa] = s[Genes.CEBPa] and not (s[Genes.GATA1] and s[Genes.FOG1] and s[Genes.SCL])
    return new_s


def _update_PU1(s):
    new_s = np.copy(s)
    new_s[Genes.PU1] = (s[Genes.CEBPa] or s[Genes.PU1]) and not (s[Genes.GATA1] or s[Genes.GATA2])
    return new_s


def _update_cJun(s):
    new_s = np.copy(s)
    new_s[Genes.cJun] = s[Genes.PU1] and not s[Genes.Gfi1]
    return new_s


def _update_EgrNab(s):
    new_s = np.copy(s)
    new_s[Genes.EgrNab] = (s[Genes.PU1] and s[Genes.cJun]) and not s[Genes.Gfi1]
    return new_s


def _update_Gfi1(s):
    new_s = np.copy(s)
    new_s[Genes.Gfi1] = s[Genes.CEBPa] and not (s[Genes.EgrNab])
    return new_s

_update_rules = {Genes.GATA2: _update_GATA2,
                 Genes.GATA1: _update_GATA1,
                 Genes.FOG1: _update_FOG1,
                 Genes.EKLF: _update_EKLF,
                 Genes.Fli1: _update_Fli1,
                 Genes.SCL: _update_SCL,
                 Genes.CEBPa: _update_CEBPa,
                 Genes.PU1: _update_PU1,
                 Genes.cJun: _update_cJun,
                 Genes.EgrNab: _update_EgrNab,
                 Genes.Gfi1: _update_Gfi1}


def update(s, gene = None):
    """
    update a specified gene or all genes from current state
    :param s: current state
    :param gene: a gene (asynchronous) or None for all the genes in the network (synchronous)
    :return: the successor state
    """
    if gene is not None:
        return _update_rules[gene](s)
    else:
        new_s = np.copy(s)
        for g in Genes:
            new_s[g] = _update_rules[g](s)[g]
        return new_s
