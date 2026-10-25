from tulip_qp.params import NetworkParams, TrainParams, SolveParams, Params


def test_network_params_defaults():
    p = NetworkParams()
    assert p.width == 256
    assert p.depth == 4


def test_train_params_defaults():
    p = TrainParams()
    assert p.epochs     == 1000
    assert p.batch_size == 64
    assert p.lr         == 1e-3
    assert p.alpha      == 1.0
    assert p.tol_off    == 1e-4
    assert p.log_every  == 50


def test_solve_params_defaults():
    p = SolveParams()
    assert p.tol_on       == 1e-4
    assert p.backup_steps == 100
    assert p.backup_lr    == 1e-3


def test_params_hierarchy():
    p = Params()
    assert isinstance(p.Network, NetworkParams)
    assert isinstance(p.Train,   TrainParams)
    assert isinstance(p.Solve,   SolveParams)


def test_params_mutable():
    p = Params()
    p.Network.width = 512
    assert p.Network.width == 512
    p.Train.lr = 5e-4
    assert p.Train.lr == 5e-4
    p.Train.alpha = 2.0
    assert p.Train.alpha == 2.0