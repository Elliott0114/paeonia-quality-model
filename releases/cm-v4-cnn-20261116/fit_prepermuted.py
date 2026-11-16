def fit(train_x, train_y, train_ids, valid_x, valid_y, folder, seed, epochs, role):
    """The outer fit never receives outer test features or target values."""
    if verify_completed(folder):
        return
    if folder.exists():
        raise RuntimeError(f'Incomplete fit preserved. Inspect before recovery: {folder}')
    folder.mkdir(parents=True)
    setup(seed)
    start = time.perf_counter()
    norm = normalization(train_x, train_y)
    expanded, lineage = augment(train_x, train_ids, seed)
    source_rows = lineage.source_row.to_numpy(int)
    x = torch.tensor((expanded - norm['input_mean']) / norm['input_std'], dtype=torch.float32, device='mps')
    zy = (train_y[source_rows] - norm['target_mean']) / norm['target_std']
    mask = torch.tensor(np.isfinite(zy), dtype=torch.float32, device='mps')
    y = torch.tensor(np.nan_to_num(zy, nan=0.), dtype=torch.float32, device='mps')
    fraction = mask.mean(0)
    assert (np.isfinite(train_y).sum(0) > 0).all()
    if valid_x is not None:
        vx = torch.tensor((valid_x - norm['input_mean']) / norm['input_std'], dtype=torch.float32, device='mps')
        vy0 = (valid_y - norm['target_mean']) / norm['target_std']
        vm = torch.tensor(np.isfinite(vy0), dtype=torch.float32, device='mps')
        vy = torch.tensor(np.nan_to_num(vy0, nan=0.), dtype=torch.float32, device='mps')
        valid_counts = np.isfinite(vy0).sum(0)
        assert (valid_counts > 0).all()
    model = build_model().to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'], fused=True)
    bs = CONFIG['batch_size']
    steps = math.ceil(len(x) / bs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['learning_rate'],
        epochs=CONFIG['max_epochs'], steps_per_epoch=steps)
    shuffle = np.random.default_rng(seed_for(seed, 'shuffle'))
    history = []
    w.write_csv(lineage, folder / 'augmentation_lineage.csv.gz')
    np.savez_compressed(folder / 'normalization.npz', **norm, train_ids=np.asarray(train_ids, str),
        observed_counts=np.isfinite(train_y).sum(0))
    dump(folder / 'fit.json', dict(role=role, seed=seed, epochs=epochs,
        original_train_n=len(train_x), expanded_train_n=len(x), validation_n=0 if valid_x is None else len(valid_x),
        target_counts=np.isfinite(train_y).sum(0).tolist(), train_ids=list(train_ids),
        started_at=now(), outer_test_access_during_fit=False))
    for epoch in range(1, epochs + 1):
        model.train()
        order = shuffle.permutation(len(x))
        epoch_index = torch.tensor(order, device="mps")
        epoch_x, epoch_y, epoch_mask = x[epoch_index], y[epoch_index], mask[epoch_index]
        loss_sum = torch.zeros((), device='mps')
        for off in range(0, len(x), bs):
            indices = slice(off, off+bs)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_huber(model(epoch_x[indices]), epoch_y[indices], epoch_mask[indices], fraction)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * len(epoch_x[indices])
        item = dict(epoch=epoch, train_huber=float((loss_sum / len(x)).cpu()),
            learning_rate=optimizer.param_groups[0]['lr'])
        if valid_x is not None:
            model.eval()
            with torch.no_grad():
                vloss = (F.huber_loss(model(vx), vy, reduction='none') * vm).sum(0).cpu().numpy()
            item['validation_huber'] = float(np.mean(vloss / valid_counts))
            for j, short in enumerate(CORE):
                item['val_sum_' + short] = float(vloss[j])
                item['val_n_' + short] = int(valid_counts[j])
        assert np.isfinite(list(item.values())).all(), (folder, epoch, item)
        history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            dump(folder / 'progress.json', dict(epoch=epoch, epochs=epochs, seconds=time.perf_counter()-start, time=now()))
    torch.mps.synchronize()
    w.write_csv(pd.DataFrame(history), folder / 'history.csv')
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=state, norm={k:torch.tensor(v) for k,v in norm.items()},
        targets=CORE, epoch=epochs, seed=seed), folder / 'model.pt')
    train_pred = predict(model, train_x, norm)
    np.savez_compressed(folder / 'training_predictions.npz', prediction=train_pred,
        truth=train_y, ids=np.asarray(train_ids, str))
    del model, optimizer, scheduler, x, y, mask
    torch.mps.empty_cache()
    hashes = {p.name:w.digest(p) for p in folder.iterdir() if p.is_file() and p.name not in ['progress.json', 'complete.json']}
    dump(folder / 'complete.json', dict(complete=True, elapsed_seconds=time.perf_counter()-start,
        completed_at=now(), hashes=hashes))
    print(f'{folder.parent.name}/{folder.name}: {epochs} epochs, {time.perf_counter()-start:.1f}s', flush=True)
