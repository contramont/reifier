"""Stage 2: a grid names every feature at every depth, and check_grid holds it to the trace."""

import copy

import pytest

from reifier import ONE, Grid, add, check_grid, held, layout, parity, trace, xor


def test_layout_puts_each_mlp_after_what_it_reads():
    t = trace(add, 4, 4)
    g = layout(t)
    assert g.rows[0] == [ONE, *t.inputs, *[None] * (len(g.rows[0]) - 9)]
    assert [len(mlps) for mlps in check_grid(t, g)] == [0, 6, 7, 6, 7, 2]  # the gates by depth
    assert len(g.rows[0]) == 16 < len(t.mlps)  # features are used again
    assert len(layout(trace(lambda a, b: add(a, b, parity), 4, 4)).rows) == 5  # a layer shallower


def test_a_grid_prints_what_each_feature_holds():
    assert str(layout(trace(xor, 3))).splitlines() == [
        "feature  depth 0  depth 1       depth 2",
        '0        ONE      "             "',
        "1        xs0      xor/atleast3  xor/odd",
        "2        xs1      xor/atleast2  .",
        "3        xs2      xor/atleast1  .",
    ]


def test_held_keeps_a_signal_until_it_is_written_over():
    t = trace(xor, 3)
    g = held(layout(t))
    assert [row[2] for row in g.rows] == ["xs1", "xor/atleast2", "xor/atleast2"]
    assert g.rows[2][1] == "xor/odd" and check_grid(t, g)


def test_row0_and_pins_place_signals():
    t = trace(xor, 2)
    g = layout(t, [ONE, "mine", *t.inputs], {"xor/odd": 1})
    assert [row[1] for row in g.rows] == ["mine", "mine", "xor/odd"] and check_grid(t, g)
    two = {"xor/atleast1": 1, "xor/odd": 1}  # then: an input; over an input; nowhere; over ONE
    for pin in (two, {"xs0": 1}, {"xor/odd": 2}, {"xor/odd": 9}, {"xor/odd": -4}, {"xor/odd": 0}):
        with pytest.raises(ValueError, match="cannot take feature"):
            layout(t, [ONE, "mine", *t.inputs], pin)


@pytest.mark.parametrize(
    "fault, message",
    [
        (lambda rows: rows[1].__setitem__(0, None), "not ONE"),
        (lambda rows: rows[1].__setitem__(1, rows[1][2]), r"depth 1 holds \['a1'\] twice"),
        (lambda rows: rows[2].__setitem__(1, None), "depth 2 lacks .'a0'."),
        (lambda rows: rows[3].__setitem__(1, "what"), "what first appears at depth 3, and no MLP"),
        (lambda rows: rows[0].__setitem__(1, None), "input a0 is not at depth 0"),
        (lambda rows: rows[4].__setitem__(2, None), "output add/xor/odd is not at the last depth"),
        (lambda rows: rows[3].__setitem__(1, "a1"), "a1 is at depth 3 but not at depth 2"),
        (lambda rows: rows[2].__setitem__(6, None), "does not hold .'add/or.2'., which add/or"),
        (lambda rows: rows[0].__setitem__(5, "add/and"), "add/and: its outputs first appear at"),
    ],
)
def test_check_grid_names_what_is_wrong(fault, message):
    t = trace(add, 2, 2)
    rows = copy.deepcopy(layout(t).rows)
    assert check_grid(t, Grid(rows))
    fault(rows)
    with pytest.raises(AssertionError, match=message):
        check_grid(t, Grid(rows))
