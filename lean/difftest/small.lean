import ReifierProofs.Transformer

namespace Reifier.DiffTest

def xor3 : Trace where
  inputs := [0, 1, 2]
  mlps := [
  { ins := [0, 1, 2], outs := [3], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 1, 2], outs := [4], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 1, 2], outs := [5], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [3, 4, 5], outs := [6], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [6]

def add22 : Trace where
  inputs := [0, 1, 2, 3]
  mlps := [
  { ins := [1, 3], outs := [4], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [], outs := [5], act := .step, hidden := 1, wi := fun j k => ([[]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [4], outs := [6], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 3, 5], outs := [7], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 3, 5], outs := [8], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 3, 5], outs := [9], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [7, 8, 9], outs := [10], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 2, 6], outs := [11], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 2, 6], outs := [12], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 2, 6], outs := [13], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [11, 12, 13], outs := [14], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [14, 10]

def add33 : Trace where
  inputs := [0, 1, 2, 3, 4, 5]
  mlps := [
  { ins := [1, 4], outs := [6], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5], outs := [7], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5, 6], outs := [8], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 4], outs := [9], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [], outs := [10], act := .step, hidden := 1, wi := fun j k => ([[]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [7], outs := [11], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [8, 9], outs := [12], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5, 10], outs := [13], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5, 10], outs := [14], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5, 10], outs := [15], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [13, 14, 15], outs := [16], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 4, 11], outs := [17], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 4, 11], outs := [18], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 4, 11], outs := [19], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [17, 18, 19], outs := [20], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 3, 12], outs := [21], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 3, 12], outs := [22], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 3, 12], outs := [23], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [21, 22, 23], outs := [24], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [24, 20, 16]

def gluadd33 : Trace where
  inputs := [0, 1, 2, 3, 4, 5]
  mlps := [
  { ins := [1, 4], outs := [6], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5], outs := [7], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5, 6], outs := [8], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-3 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 4], outs := [9], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [], outs := [10], act := .step, hidden := 1, wi := fun j k => ([[]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [7], outs := [11], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [8, 9], outs := [12], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 5, 10], outs := [13], act := .reglu, hidden := 3, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(4 : ℚ), (4 : ℚ), (4 : ℚ)]].getD j []).getD k 0, b := fun j => [((-1 : ℚ) / 2), ((-5 : ℚ) / 2), ((-3 : ℚ) / 2), (6 : ℚ), (14 : ℚ), (-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1, 4, 11], outs := [14], act := .reglu, hidden := 3, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(4 : ℚ), (4 : ℚ), (4 : ℚ)]].getD j []).getD k 0, b := fun j => [((-1 : ℚ) / 2), ((-5 : ℚ) / 2), ((-3 : ℚ) / 2), (6 : ℚ), (14 : ℚ), (-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 3, 12], outs := [15], act := .reglu, hidden := 3, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(4 : ℚ), (4 : ℚ), (4 : ℚ)]].getD j []).getD k 0, b := fun j => [((-1 : ℚ) / 2), ((-5 : ℚ) / 2), ((-3 : ℚ) / 2), (6 : ℚ), (14 : ℚ), (-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [15, 14, 13]

def odd : Trace where
  inputs := [0, 1]
  mlps := [
  { ins := [1], outs := [2], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 0], outs := [3], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2, 0], outs := [4], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [3, 4], outs := [5], act := .step, hidden := 1, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0, 1, 0], outs := [6, 7], act := .step, hidden := 2, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ)], [(0 : ℚ), (1 : ℚ), (0 : ℚ)]].getD j []).getD k 0, b := fun j => [(-2 : ℚ), (-1 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ), (0 : ℚ)], [(0 : ℚ), (1 : ℚ)]].getD i []).getD j 0 },
  { ins := [], outs := [8], act := .step, hidden := 1, wi := fun j k => ([[]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [1, 8, 7, 5]

def par4 : Trace where
  inputs := [0, 1, 2, 3]
  mlps := [
  { ins := [0, 1, 2, 3], outs := [4], act := .reglu, hidden := 4, wi := fun j k => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)], [(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(-4 : ℚ), (-4 : ℚ), (-4 : ℚ), (-4 : ℚ)], [(4 : ℚ), (4 : ℚ), (4 : ℚ), (4 : ℚ)], [(4 : ℚ), (4 : ℚ), (4 : ℚ), (4 : ℚ)]].getD j []).getD k 0, b := fun j => [((-1 : ℚ) / 2), ((-5 : ℚ) / 2), ((-3 : ℚ) / 2), ((-7 : ℚ) / 2), (6 : ℚ), (14 : ℚ), (-2 : ℚ), (-10 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [4]

def soa : Trace where
  inputs := [0, 1, 2]
  mlps := [
  { ins := [0, 1, 2], outs := [3], act := .reglu, hidden := 2, wi := fun j k => ([[(1 : ℚ), (-1 : ℚ), (0 : ℚ)], [(0 : ℚ), (1 : ℚ), (1 : ℚ)], [(-4 : ℚ), (4 : ℚ), (0 : ℚ)], [(0 : ℚ), (-4 : ℚ), (-4 : ℚ)]].getD j []).getD k 0, b := fun j => [((-1 : ℚ) / 2), ((-3 : ℚ) / 2), (6 : ℚ), (10 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ), (1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [3]

def chain : Trace where
  inputs := [0, 1, 2, 3]
  mlps := [
  { ins := [3], outs := [4], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [0], outs := [5], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [1], outs := [6], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [2], outs := [7], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [7], outs := [8], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [4], outs := [9], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [5], outs := [10], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [6], outs := [11], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [11], outs := [12], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [8], outs := [13], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [9], outs := [14], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [10], outs := [15], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [15], outs := [16], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [12], outs := [17], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [13], outs := [18], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [14], outs := [19], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [19], outs := [20], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [16], outs := [21], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [17], outs := [22], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 },
  { ins := [18], outs := [23], act := .step, hidden := 1, wi := fun j k => ([[(-1 : ℚ)]].getD j []).getD k 0, b := fun j => [(0 : ℚ)].getD j 0, wo := fun i j => ([[(1 : ℚ)]].getD i []).getD j 0 }]
  outputs := [20, 21, 22, 23]

def cellStr : Option Name → String
  | none => "null"
  | some .one => "[\"one\"]"
  | some (.code i) => s!"[\"code\", {i}]"
  | some (.sig s) => s!"[\"sig\", {s}]"

def arr (xs : List String) : String := "[" ++ ", ".intercalate xs ++ "]"

def gridJson (g : Grid) : String := arr (g.rows.map fun r => arr ((List.range g.width).map (cellStr ∘ r)))

def qs (x : ℚ) : String := "\"" ++ toString x ++ "\""

/-- `emit`'s layers, computably (`one` is a real, so `emit` is not). -/
def layersOf (t : Trace) (g : Grid) : List (List Hidden) :=
  (g.rows.zip g.rows.tail).map fun br => layerUnits g.width t.mlps br.1 br.2

theorem layersOf_eq (t : Trace) (g : Grid) : layersOf t g = (emit t g).layers := rfl

def layersJson (t : Trace) : String :=
  let g := tokenGrid t
  arr ((layersOf t g).map fun us => arr (us.map fun u =>
    arr [arr ((List.range g.width).map (qs ∘ u.gate)), arr ((List.range g.width).map (qs ∘ u.val)),
         arr ((List.range g.width).map (qs ∘ u.out))]))

def progJson (t : Trace) : String :=
  "{\"layout\": " ++ gridJson (layout t) ++ ", \"token\": " ++ gridJson (tokenGrid t) ++ ", \"layers\": " ++ layersJson t ++ "}"

def allJson : String := "{" ++ ", ".intercalate ["\"xor3\": " ++ progJson xor3, "\"add22\": " ++ progJson add22, "\"add33\": " ++ progJson add33, "\"gluadd33\": " ++ progJson gluadd33, "\"odd\": " ++ progJson odd, "\"par4\": " ++ progJson par4, "\"soa\": " ++ progJson soa, "\"chain\": " ++ progJson chain] ++ "}"

#eval IO.FS.writeFile "lean_small.json" allJson

end Reifier.DiffTest
