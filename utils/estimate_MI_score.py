import numpy as np
import torch

    
def Distance_cal(data1, data2, return_sq = False, run_gpu=-1):
    n1, dim = data1.shape
    n2 = data2.shape[0]
    if run_gpu < 0:
        run_gpu = data1.device.index
    dev = 'cuda:' + str(run_gpu)
    dists = torch.zeros([n1,n2], device=dev)
    if run_gpu != data1.device.index:
        run_cuda = 'cuda:' + str(run_gpu)
        for id in range(dim):
            dists += ((data1[:,id].reshape([-1,1]) - data2[:,id].reshape([1,-1]))**2).to(run_cuda)
    else:
        for id in range(dim):
            dists += (data1[:,id].reshape([-1,1]) - data2[:,id].reshape([1,-1]))**2
    if return_sq:
        return dists
    return torch.sqrt(dists)

# get the indexes to the nearest neighbors (from data1 to (data1 and data2)) up to 20th
def get_nns_idxes(data1, data2, num_k1=20, num_k2=20, mc_chunk_size=10000, run_gpu=-1):
    # nearest neighbors

    datanum1 = len(data1)
    datanum2 = len(data2)
    num_k1 = np.min([num_k1, datanum1 - 1])
    num_k2 = np.min([num_k2, datanum2])
    mc_chunk_size = np.min([mc_chunk_size, datanum1])

    mc_datanum = datanum1
    mc_data = data1
    target_data2 = data2

    cur_dev = data1.get_device()
    nn_i1s_append = torch.tensor([], device=cur_dev)
    nn_i2s_append = torch.tensor([], device=cur_dev)
    for imc in range(int((mc_datanum - 1)/mc_chunk_size) + 1):
    #     print(imc, imc*mc_chunk_size, (imc + 1)*mc_chunk_size)
        tst_idxes = np.arange(imc*mc_chunk_size, \
                              np.min([(imc + 1)*mc_chunk_size, mc_datanum]))
        dists = Distance_cal(mc_data[tst_idxes], mc_data, run_gpu=run_gpu)
        # get k1 + 1 nearest neighbors and throw away the first nearest neighbor
        nn_i1s = torch.topk(dists, k = num_k1+1, axis=1, largest=False, sorted=True).indices
        nn_i1s_append = torch.cat([nn_i1s_append, nn_i1s[:,1:]], dim=0)
        # get k2 nearest neighbors
        dists = Distance_cal(mc_data[tst_idxes], target_data2, run_gpu=run_gpu)
        nn_i2s = torch.topk(dists, k = num_k2, axis=1, largest=False, sorted=True).indices
        nn_i2s_append = torch.cat([nn_i2s_append, nn_i2s], dim=0)

    return nn_i1s_append, nn_i2s_append

# get nearest neighbor distances (from data1 to data2) up to 20th
def get_nn_idxes_self(data1, num_k=20, mc_chunk_size=10000, run_gpu=-1):
    # Nearest neighbor
    datanum1 = len(data1)
    num_k = np.min([num_k, datanum1 - 1])
    mc_chunk_size = np.min([mc_chunk_size, datanum1])

    mc_datanum = datanum1
    mc_data = data1
    target_data = data1

    dev = data1.get_device()
    nn_i1s_append = torch.tensor([], device=dev)
    for imc in range(int((mc_datanum - 1)/mc_chunk_size) + 1):
        tst_idxes = np.arange(imc*mc_chunk_size, \
                              np.min([(imc + 1)*mc_chunk_size, mc_datanum]))
        # get k nearest neighbors
        dists = Distance_cal(mc_data[tst_idxes], target_data, run_gpu=run_gpu)
#         print(dists)
        nn_i1s = torch.topk(dists, k = num_k+1, axis=1, largest=False, sorted=True).indices
        nn_i1s_append = torch.cat([nn_i1s_append, nn_i1s[:,1:]], dim=0)
    return nn_i1s_append

# average of estimators for various ks
def entropy_ks_torch(data1, min_k=1, max_k=3, Nmid_k=0, nn_i1s=[], run_gpu=-1):
    if run_gpu < 0:
        run_gpu = data1.device.index
    if len(data1.shape) == 1:
        data1 = data1.reshape([-1,1])
    dim = data1.shape[1]
    mc_datanum = len(data1)
    if len(nn_i1s) == 0:
        nn_i1s = get_nn_idxes_self(data1, max_k, run_gpu=run_gpu)  # required_grad == False

    dev = 'cuda:' + str(run_gpu)
    loggamma = dim/2.*torch.log(torch.tensor(torch.pi)) - torch.special.gammaln(torch.tensor(dim/2. + 1.))
    
    ent_est = torch.tensor(0., device=dev) # initialize
    if Nmid_k < 0:
        Nmid_k = 0
    if min_k == max_k:
        Nmid_k = -1
    ks = np.linspace(min_k-1, max_k-1, Nmid_k+2, dtype=int)
    for ik in ks:
        nn_d1s = torch.sqrt(torch.sum((data1 - data1[nn_i1s[:,ik].long()])**2, axis=1))
        ent_est += dim*torch.sum(torch.log(nn_d1s))/mc_datanum - torch.digamma(torch.tensor(ik+1))
    ent_est = ent_est/len(ks) + loggamma + torch.log(torch.tensor(mc_datanum-1))
    return ent_est, nn_i1s

def KL_div_ks_torch(data1, data2, min_k=1, max_k=3, Nmid_k=0, nn_i1s=[], nn_i2s=[], return_nns=True, run_gpu=-1, **ignore):
    if len(data1.shape) == 1:
        data1 = data1.reshape([-1,1])
    if len(data2.shape) == 1:
        data2 = data2.reshape([-1,1])
    dim = data1.shape[1]
    datanum1 = len(data1)
    datanum2 = len(data2)
    mc_datanum = len(data1)
    min_k=min(min_k, datanum1-1, datanum2)
    max_k=min(max_k, datanum1-1, datanum2)
    if run_gpu < 0:
        run_gpu = data1.device.index
    if len(nn_i1s) == 0:
        # get_nns_idxes 함수가 max_k+1 이 아니라 max_k 를 요구할 수 있으니 확인 필요
        nn_i1s, nn_i2s = get_nns_idxes(data1, data2, max_k, max_k, run_gpu=run_gpu) # k번째 이웃까지 필요

    # dev = 'cuda:' + str(run_gpu) # 아래에서 data1.device 사용으로 대체 가능
    current_device = data1.device # 입력 데이터와 동일한 장치 사용
    KL_est = torch.tensor(0., device=current_device) # initialize

    # --- Epsilon 추가 ---
    eps = 1e-9 # 수치적 안정성을 위한 작은 값 (1e-8 ~ 1e-10 사이에서 조정 가능)

    if Nmid_k < 0: Nmid_k = 0
    if min_k == max_k: Nmid_k = -1
    # k 값 범위 확인: indices는 0부터 시작하므로 k-1 ~ max_k-1
    ks = np.linspace(min_k-1, max_k-1, Nmid_k+2, dtype=int)
    valid_k_count = 0 # 유효한 k 개수 카운트

    for ik in ks:
        # k 값 유효성 재확인 (nn_i1s, nn_i2s의 크기보다 작아야 함)
        if ik >= nn_i1s.shape[1] or ik >= nn_i2s.shape[1]:
             print(f"Warning: Skipping k index {ik} (k={ik+1}) as it exceeds nn index dimensions ({nn_i1s.shape[1]}, {nn_i2s.shape[1]})")
             continue

        try:
            # 제곱 거리 계산
            sq_dist1 = torch.sum((data1 - data1[nn_i1s[:,ik].long()])**2, axis=1)
            sq_dist2 = torch.sum((data1 - data2[nn_i2s[:,ik].long()])**2, axis=1)

            # --- Epsilon 추가 ---
            # 제곱근 계산 전에 epsilon 더하기
            nn_d1s = torch.sqrt(sq_dist1 + eps)
            nn_d2s = torch.sqrt(sq_dist2 + eps)

            # --- Epsilon 추가 (선택 사항, 추가 안정성) ---
            # 로그 계산 전에 epsilon 더하기
            log_u1s = torch.log(nn_d1s + eps)
            log_u2s = torch.log(nn_d2s + eps)

            # NaN/Inf 발생 시 처리 (Epsilon 추가 후 발생 가능성 낮아짐)
            term = log_u2s - log_u1s
            if torch.isnan(term).any() or torch.isinf(term).any():
                print(f"Warning: NaN/Inf detected in log term for k={ik+1}. Clamping values.")
                term = torch.nan_to_num(term, nan=0.0, posinf=100.0, neginf=-100.0) # 예시: 큰 값으로 제한

            KL_est += torch.sum(term)
            valid_k_count += 1 # 유효하게 계산된 k 증가

        except IndexError as e:
             print(f"Error accessing nearest neighbor indices for k index {ik} (k={ik+1}): {e}")
             continue # 다음 k로 넘어감

    # 유효한 k가 하나도 없으면 0 반환 (또는 에러 처리)
    if valid_k_count == 0:
        print("Warning: No valid k values processed in KL divergence estimation. Returning 0.")
        if return_nns: return torch.tensor(0., device=current_device), nn_i1s, nn_i2s
        else: return torch.tensor(0., device=current_device)

    # 최종 계산 시 유효한 k 개수로 나누기
    KL_est = dim * KL_est / mc_datanum / valid_k_count + torch.log(torch.tensor(datanum2, device=current_device)) - torch.log(torch.tensor(datanum1 - 1, device=current_device))

    if return_nns:
        return KL_est, nn_i1s, nn_i2s
    return KL_est

def MI_from_divergence(feature1, feature2, divergence_func, n_permute=1, return_nns=False, indices = {}, 
                       run_gpu=-1, *args, **kwargs):
    # I(X;Y) = KL(p(x,y)||p(x)p(y))
    assert len(feature1) == len(feature2)
    if run_gpu < 0:
        run_gpu = feature1.device.index
    dev = 'cuda:' + str(run_gpu)
    if len(feature1.shape) == 1:
        feature1 = feature1.reshape([-1,1])
    if len(feature2.shape) == 1:
        feature2 = feature2.reshape([-1,1])
    
    data1 = torch.cat([feature1, feature2], dim=1)
    MI_est = torch.tensor(0., device=dev) # initialize
    if not indices or len(indices['permute']) != n_permute or len(indices['nn'][0]) != n_permute or len(indices['nn'][1]) != n_permute:
        indices = {'nn': [[], []], 'permute': []}
    
    for i in range(n_permute):
        if len(indices['permute']) == i:
            idx = torch.randperm(len(feature2))
            indices['permute'].append(idx)
        elif len(indices['permute']) == n_permute:
            idx = indices['permute'][i]
        else:
            raise NotImplementedError
        
        data2 = torch.cat([feature1, feature2[idx]], dim=1)
        if return_nns:
            if len(indices['nn'][0]) == i and len(indices['nn'][1]) == i :
                KL_est, nn_i1s, nn_i2s = divergence_func(data1, data2, return_nns=True, *args, **kwargs)
                indices['nn'][0].append(nn_i1s)
                indices['nn'][1].append(nn_i2s)
            elif len(indices['nn'][0]) == n_permute and len(indices['nn'][1]) == n_permute:
                KL_est, _, _ = divergence_func(data1, data2, return_nns=True, 
                                               nn_i1s=indices['nn'][0][i], nn_i2s=indices['nn'][1][i], *args, **kwargs)
            else:
                raise NotImplementedError
        else:
            KL_est = divergence_func(data1, data2, return_nns=False, *args, **kwargs)
        print(KL_est)
        MI_est += KL_est
    print("MI_est: ", MI_est)
    print("MI_est / n_permute: ", MI_est / n_permute)
    if return_nns:
        return MI_est / n_permute, indices
    return MI_est / n_permute

def MI_from_divergence_v2(feature1, feature2, divergence_func, n_permute=1, fix_data1=False, return_nns=False, indices = {}, 
                       run_gpu=-1, *args, **kwargs):
    # I(X;Y) = KL(p(x,y)||p(x)p(y))
    assert len(feature1) == len(feature2)
    if run_gpu < 0:
        run_gpu = feature1.device.index
    dev = 'cuda:' + str(run_gpu)
    N = len(feature1)
    if len(feature1.shape) == 1:
        feature1 = feature1.reshape([-1,1])
    if len(feature2.shape) == 1:
        feature2 = feature2.reshape([-1,1])
    MI_est = torch.tensor(0., device=dev) # initialize
    
    if not indices or len(indices['permute1']) != n_permute or len(indices['permute2']) != n_permute or len(indices['nn'][0]) != n_permute or len(indices['nn'][1]) != n_permute:
        indices = {'nn': [[], []], 'permute1': [], 'permute2': []}
    
    for i in range(n_permute):
        if len(indices['permute1']) == i:
            if fix_data1:
                idx = torch.arange(len(feature2))
            else:
                idx = torch.randperm(len(feature2))
            indices['permute1'].append(idx)
        elif len(indices['permute1']) == n_permute:
            idx = indices['permute1'][i]
        else:
            raise NotImplementedError
        if len(indices['permute2']) == i:
            idx2 = torch.randperm(len(idx[N//2:]))
            indices['permute2'].append(idx2)
        elif len(indices['permute2']) == n_permute:
            idx2 = indices['permute2'][i]
        else:
            raise NotImplementedError
        data1 = torch.cat([feature1[idx[:N//2]], feature2[idx[:N//2]]], dim=1)
        data2 = torch.cat([feature1[idx[N//2:]], feature2[idx[N//2:]][idx2]], dim=1)
        
        if return_nns:
            if len(indices['nn'][0]) == i and len(indices['nn'][1]) == i :
                KL_est, nn_i1s, nn_i2s = divergence_func(data1, data2, return_nns=True, *args, **kwargs)
                indices['nn'][0].append(nn_i1s)
                indices['nn'][1].append(nn_i2s)
            elif len(indices['nn'][0]) == n_permute and len(indices['nn'][1]) == n_permute:
                KL_est, _, _ = divergence_func(data1, data2, return_nns=True, 
                                               nn_i1s=indices['nn'][0][i], nn_i2s=indices['nn'][1][i], *args, **kwargs)
            else:
                raise NotImplementedError
        else:
            KL_est = divergence_func(data1, data2, return_nns=False, *args, **kwargs)
        
        MI_est += KL_est
    if return_nns:
        return MI_est / n_permute, indices
    return MI_est / n_permute
