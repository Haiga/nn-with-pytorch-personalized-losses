import torch


def zRisk(mat, alpha, requires_grad=False):
    alpha = torch.tensor([alpha], requires_grad=requires_grad, dtype=torch.float)
    si = torch.sum(mat[:, 0])
    tj = torch.sum(mat, dim=1)
    n = torch.sum(tj)

    xij_eij = mat[:, 0] - si * (tj / n)
    den = torch.sqrt(si * (tj / n))
    div = xij_eij / den

    less0 = (mat[:, 0] - si * (tj / n)) / (torch.sqrt(si * (tj / n))) < 0

    less0 = alpha * less0
    z_risk = div * less0 + div
    # z_risk[torch.isnan(z_risk)] = 0
    z_risk = torch.sum(z_risk)

    return z_risk


def zRiskRemoveNan(mat, alpha, requires_grad=False):
    alpha = torch.tensor([alpha], requires_grad=requires_grad, dtype=torch.float)
    si = torch.sum(mat[:, 0])
    tj = torch.sum(mat, dim=1)
    n = torch.sum(tj)

    xij_eij = mat[:, 0] - si * (tj / n)
    den = torch.sqrt(si * (tj / n))
    den[torch.isnan(den)] = 0
    div = xij_eij / den
    less0 = (mat[:, 0] - si * (tj / n)) / (torch.sqrt(si * (tj / n))) < 0
    less0 = alpha * less0
    z_risk = div * less0 + div
    z_risk[torch.isnan(z_risk)] = 0
    z_risk = torch.sum(z_risk)

    return z_risk


def geoRisk(mat, alpha, requires_grad=False):
    si = torch.sum(mat[:, 0])
    z_risk = zRisk(mat, alpha, requires_grad=requires_grad)

    num_queries = mat.shape[0]
    value = z_risk / num_queries
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncd = m.cdf(value)
    return torch.sqrt((si / num_queries) * ncd)

def maxGeoRisk(mat, alpha, requires_grad=False):
    mat[:, 0] = torch.max(mat, dim=1)[0]
    si = torch.sum(mat[:, 0])
    z_risk = zRisk(mat, alpha, requires_grad=requires_grad)

    num_queries = mat.shape[0]
    value = z_risk / num_queries
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncd = m.cdf(value)
    return torch.sqrt((si / num_queries) * ncd)

def zRiskAtI(mat, alpha, requires_grad=False, i=0):
    alpha = torch.tensor([alpha], requires_grad=requires_grad, dtype=torch.float)
    si = torch.sum(mat[:, i])
    tj = torch.sum(mat, dim=1)
    n = torch.sum(tj)

    xij_eij = mat[:, i] - si * (tj / n)
    den = torch.sqrt(si * (tj / n))
    div = xij_eij / den

    less0 = (mat[:, i] - si * (tj / n)) / (torch.sqrt(si * (tj / n))) < 0

    less0 = alpha * less0
    z_risk = div * less0 + div
    # z_risk[torch.isnan(z_risk)] = 0
    z_risk = torch.sum(z_risk)

    return z_risk


def allGeoRisk(mat, alpha, requires_grad=False):
    all_geos = []
    for i in range(mat.shape[1]):
        si = torch.sum(mat[:, i])
        z_risk = zRiskAtI(mat, alpha, requires_grad=requires_grad, i=i)

        num_queries = mat.shape[0]
        value = z_risk / num_queries
        m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        ncd = m.cdf(value)
        g = torch.sqrt((si / num_queries) * ncd)
        all_geos.append(g)
    return torch.stack(all_geos)


def geoRiskRemoveNan(mat, alpha, requires_grad=False):
    si = torch.sum(mat[:, 0])
    z_risk = zRiskRemoveNan(mat, alpha, requires_grad=requires_grad)

    num_queries = mat.shape[0]
    value = z_risk / num_queries
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncd = m.cdf(value)
    return torch.sqrt((si / num_queries) * ncd)
