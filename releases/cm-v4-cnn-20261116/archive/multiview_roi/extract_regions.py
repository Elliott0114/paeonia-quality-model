"""Outcome-blind eight-region summaries from frozen ROI masks and read-only cubes."""
from common import *
import time

def partition(mask):
    coords=np.argwhere(mask)
    assert len(coords)>=8
    def split(idx,depth):
        if depth==0:return [idx]
        c=coords[idx];span=np.ptp(c,axis=0);axis=int(np.argmax(span))
        order=np.lexsort((c[:,1-axis],c[:,axis]));ordered=idx[order];cut=len(idx)//2
        return split(ordered[:cut],depth-1)+split(ordered[cut:],depth-1)
    groups=split(np.arange(len(coords)),3)
    assert np.array_equal(np.sort(np.concatenate(groups)),np.arange(len(coords)))
    return coords,groups

def main():
    check_sources();assert not (OUT/'inputs/regions.npz').exists()
    ps=source_paths();helper=import_file('roi_readonly_helpers',ps['extractor'])
    t,x,y,ss=load();q=read(ps['hreb']/'portion_qc.csv').set_index('canonical_id');geom=read(ps['c01']/'meta/group_metadata.csv')
    base=np.load(ps['hreb']/'spectra_base.npz');waves=base['wavelengths_nm'];data=[];weights=[];info=[];paths={};start=time.time()
    for batch in sorted(t.sample_number.unique()):
        stem=f'SBS-{batch:02d}';hdr=ps['cube']/(stem+'.hdr')
        arr,h,_=helper.open_envi_memmap(hdr);wi=np.flatnonzero((np.asarray(h.wavelengths)>=430)&(np.asarray(h.wavelengths)<=970))
        np.testing.assert_array_equal(np.asarray(h.wavelengths)[wi],waves)
        for row in t[t.sample_number==batch].itertuples():
            cid=row.canonical_id;r=q.loc[cid];factor=float(r.white_exposure_us/r.sample_exposure_us);assert abs(factor-r.exposure_correction_factor)<1e-8;gr=geom[(geom.sample_number==batch)&(geom.group_index==row.group_index)].iloc[0]
            mp=ps['c01']/'groups'/gr.sample_stem/f'{gr.position}.npz'
            with np.load(mp,allow_pickle=False) as z:mask=z['mask'].astype(bool);bbox=z['bbox']
            assert mask.sum()==r.roi_pixel_count
            np.testing.assert_array_equal(bbox,[r.bbox_y0,r.bbox_y1,r.bbox_x0,r.bbox_x1])
            coords,groups=partition(mask)
            crop=helper.extract_crop_bands(arr,h,*bbox,wi).astype(float)
            vendor=np.asfortranarray(crop[mask]);values=vendor*factor
            assert np.isfinite(values).all()
            index=int(t.index[t.canonical_id.eq(cid)][0])
            np.testing.assert_array_equal((vendor.mean(0)*factor).astype(np.float32),x[index])
            regions=np.asarray([vendor[g].mean(0)*factor for g in groups],np.float32)
            wts=np.asarray([len(g)/len(coords) for g in groups],float)
            recomposed=wts@regions.astype(float)
            np.testing.assert_allclose(recomposed,x[index],rtol=2e-7,atol=2e-8)
            np.testing.assert_array_equal(((np.quantile(vendor,.75,axis=0)-np.quantile(vendor,.25,axis=0))*factor).astype(np.float32),base['exposure_adjusted_iqr'][index])
            data.append((index,regions));weights.append((index,wts))
            for j,g in enumerate(groups):
                c=coords[g];info.append(dict(canonical_id=cid,sample_number=int(batch),group_index=int(row.group_index),region=j,pixels=len(g),weight=float(wts[j]),y_min=int(c[:,0].min()+bbox[0]),y_max=int(c[:,0].max()+bbox[0]),x_min=int(c[:,1].min()+bbox[2]),x_max=int(c[:,1].max()+bbox[2]),recomposition_max_error=float(np.max(abs(recomposed-x[index])))))
            paths[cid]=dict(mask=str(mp),hdr=str(hdr),factor=factor)
            del crop,values,vendor
        del arr
        dump(OUT/'extraction_progress.json',dict(completed_batches=int(batch),total_batches=69,seconds=time.time()-start,at=now()))
        print(f'ROI extraction batch {batch}/69',flush=True)
    data=np.stack([v for i,v in sorted(data)]);weights=np.stack([v for i,v in sorted(weights)])
    assert data.shape==(207,8,262) and weights.shape==(207,8)
    np.savez_compressed(OUT/'inputs/regions.npz',canonical_id=t.canonical_id.to_numpy(str),spectra=data,weights=weights,wavelengths_nm=waves)
    write(pd.DataFrame(info),OUT/'inputs/region_geometry.csv');dump(OUT/'inputs/region_sources.json',paths)
    m=json.loads((OUT/'manifest.json').read_text())
    for n in ['regions.npz','region_geometry.csv','region_sources.json']:m['inputs'][n]=digest(OUT/'inputs'/n)
    dump(OUT/'manifest.json',m)
    dump(OUT/'qa/extraction.json',dict(passed=True,grabs=207,batches=69,regions=1656,every_pixel_once=True,original_P0_reproduced=True,existing_IQR_reproduced=True,selection_uses_labels=False,max_error=max(r['recomposition_max_error'] for r in info),seconds=time.time()-start))
    check_sources()

if __name__=='__main__':main()
