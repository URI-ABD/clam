//! Tests for PanCakes.

use core::fmt::Debug;

use rand::prelude::*;

use abd_clam::{
    Dataset, DistanceValue, Tree,
    cakes::{KnnBfs, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear},
    pancakes::{Codec, CompressiveSearch, MaybeCompressedItem},
};

#[derive(Debug, PartialEq, Eq, Clone)]
struct TestItem<const N: usize> {
    arr: [char; N],
}

impl<const N: usize> Codec for TestItem<N> {
    // A simple compression scheme that encodes each pairwise difference between characters along with the index.
    type Compressed = Vec<(u8, char)>;

    fn compress(&self, target: &Self) -> Self::Compressed {
        self.arr
            .iter()
            .zip(target.arr.iter())
            .enumerate()
            .filter_map(|(i, (&a, &b))| if a != b { Some((i as u8, b)) } else { None })
            .collect()
    }

    fn decompress(&self, compressed: &Self::Compressed) -> Self {
        let mut arr = self.arr;
        for &(i, c) in compressed {
            arr[i as usize] = c;
        }
        Self { arr }
    }

    fn compressed_size(compressed: &Self::Compressed) -> usize {
        compressed.len() * (1 + std::mem::size_of::<char>())
    }

    fn original_size(&self) -> usize {
        self.arr.len() * std::mem::size_of::<char>()
    }
}

#[test]
fn pair() {
    type Tenner = TestItem<10>;

    let item1 = Tenner { arr: ['a'; 10] };
    let item2 = Tenner {
        arr: ['a', 'c', 'a', 't', 'a', 'g', 'a', 'a', 'a', 'a'],
    };

    assert_eq!(item1.original_size(), 10 * std::mem::size_of::<char>());
    assert_eq!(item2.original_size(), 10 * std::mem::size_of::<char>());

    let compressed = item1.compress(&item2);
    assert_eq!(compressed, vec![(1, 'c'), (3, 't'), (5, 'g')]);

    let original_size = item2.original_size();
    let compressed_size = Tenner::compressed_size(&compressed);
    assert_eq!(compressed_size, 3 * (1 + std::mem::size_of::<char>()));
    assert!(
        compressed_size < original_size,
        "Compression should reduce size. Got compressed size {compressed_size} and original size {original_size}."
    );

    let decompressed = item1.decompress(&compressed);
    assert_eq!(item2, decompressed);
    assert_eq!(item2.original_size(), decompressed.original_size());
}

fn gen_test_item<R: Rng, const N: usize>(rng: &mut R, chars: &[char]) -> Result<TestItem<N>, String> {
    let mut arr = [chars[0]; N];
    for c in &mut arr {
        *c = *chars.choose(rng).ok_or("chars should not be empty".to_string())?;
    }
    Ok(TestItem { arr })
}

fn hamming<const N: usize>(a: &TestItem<N>, b: &TestItem<N>) -> usize {
    a.arr.iter().zip(b.arr.iter()).filter(|(a, b)| a != b).count()
}

fn compare_trees<Item, D, CD, M, T, A>(tree: &Tree<D, M, T, A>, compressed_tree: &Tree<CD, M, T, A>, ratio: f64) -> Result<(), String>
where
    Item: Codec + Eq + Debug,
    D: Dataset<Item = Item>,
    D::Id: Eq + Debug,
    CD: Dataset<Item = MaybeCompressedItem<Item>, Id = D::Id>,
{
    let items_size = tree.dataset().as_slice().iter().map(|(_, item)| item.original_size()).sum::<usize>();
    let n_clusters = tree.cluster_map().len();
    let compressed_size = compressed_tree.dataset().as_slice().iter().map(|(_, item)| item.size()).sum::<usize>();
    let n_clusters_compressed = compressed_tree.cluster_map().len();

    assert!(
        compressed_size < items_size,
        "Compression should reduce size of items. Got compressed size {compressed_size} and original size {items_size}."
    );
    let items_ratio = compressed_size as f64 / items_size as f64;
    assert!(
        items_ratio < ratio,
        "Items not compressed enough. Got compressed size {compressed_size} and original size {items_size}, for a ratio of {items_ratio:.2}."
    );

    assert!(
        n_clusters_compressed < n_clusters,
        "Compression should reduce number of clusters. Got {n_clusters_compressed} compressed clusters and {n_clusters} original clusters."
    );
    let clusters_ratio = n_clusters_compressed as f64 / n_clusters as f64;
    assert!(
        clusters_ratio < ratio,
        "Clusters not trimmed enough. Got {n_clusters_compressed} compressed clusters and {n_clusters} original clusters, for a ratio of {clusters_ratio:.2}."
    );

    Ok(())
}

#[test]
fn compression() -> Result<(), String> {
    let mut rng = rand::rng();
    let chars = ['a', 'c', 't', 'g'];
    let items = (0..1_000).map(|_| gen_test_item::<_, 8>(&mut rng, &chars)).collect::<Result<Vec<_>, _>>()?;
    let items = items.into_iter().enumerate().collect::<Vec<_>>();

    let tree = Tree::new_minimal(items, hamming)?;
    let compressed_tree = tree.clone().compress_all::<Vec<_>>(3);
    compare_trees(&tree, &compressed_tree, 0.75)?;

    let decompressed_tree = compressed_tree.decompress_all::<Vec<_>>();
    assert_eq!(tree.dataset(), decompressed_tree.dataset(), "Decompressed items should match original items.");

    Ok(())
}

#[test]
fn par_compression() -> Result<(), String> {
    let mut rng = rand::rng();
    let chars = ['a', 'c', 't', 'g'];
    let items = (0..100_000).map(|_| gen_test_item::<_, 10>(&mut rng, &chars)).collect::<Result<Vec<_>, _>>()?;
    let items = items.into_iter().enumerate().collect::<Vec<_>>();

    let tree = Tree::par_new_minimal(items, hamming)?;
    let compressed_tree = tree.clone().par_compress_all::<Vec<_>>(3);
    compare_trees(&tree, &compressed_tree, 0.5)?;

    let decompressed_tree = compressed_tree.par_decompress_all::<Vec<_>>();
    assert_eq!(tree.dataset(), decompressed_tree.dataset(), "Decompressed items should match original items.");

    Ok(())
}

#[test]
fn search() -> Result<(), String> {
    let mut rng = rand::rng();
    let chars = ['a', 'c', 't', 'g'];
    let data = (0..1_000).map(|_| gen_test_item::<_, 8>(&mut rng, &chars)).collect::<Result<Vec<_>, _>>()?;
    let data = data.into_iter().enumerate().collect::<Vec<_>>();
    let query = gen_test_item::<_, 8>(&mut rng, &chars)?;

    let mut tree = Tree::new_minimal(data, hamming)?.compress_all::<Vec<_>>(3);

    for radius in [1, 2, 4] {
        let linear_alg = RnnLinear(radius);
        let linear_hits = linear_alg.compressive_search(&mut tree, &query)?;
        let linear_hits = sort_nondescending(linear_hits);
        tree.compress_root();

        let chess_alg = RnnChess(radius);
        let chess_hits = chess_alg.compressive_search(&mut tree, &query)?;
        let chess_hits = sort_nondescending(chess_hits);
        tree.compress_root();

        check_hits(&linear_hits, &chess_hits, format!("RnnChess({radius})"))?;
    }

    for k in [1, 10, 20] {
        let linear_alg = KnnLinear(k);
        let linear_hits = linear_alg.compressive_search(&mut tree, &query)?;
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(tree.dataset().len()),
            "Not enough linear hits {} for k={}",
            linear_hits.len(),
            k.min(tree.dataset().len())
        );
        tree.compress_root();

        let dfs_alg = KnnDfs(k);
        let dfs_hits = dfs_alg.compressive_search(&mut tree, &query)?;
        let dfs_hits = sort_nondescending(dfs_hits);
        tree.compress_root();
        check_hits(&linear_hits, &dfs_hits, format!("KnnDfs({k})"))?;

        let bfs_alg = KnnBfs(k);
        let bfs_hits = bfs_alg.compressive_search(&mut tree, &query)?;
        let bfs_hits = sort_nondescending(bfs_hits);
        tree.compress_root();
        check_hits(&linear_hits, &bfs_hits, format!("KnnBfs({k})"))?;

        let rrnn_alg = KnnRrnn(k);
        let rrnn_hits = rrnn_alg.compressive_search(&mut tree, &query)?;
        let rrnn_hits = sort_nondescending(rrnn_hits);
        tree.compress_root();
        check_hits(&linear_hits, &rrnn_hits, format!("KnnRrnn({k})"))?;
    }

    Ok(())
}

fn sort_nondescending(mut items: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    items.sort_by_key(|(_, d)| *d);
    items
}

fn check_hits<T: DistanceValue>(expected: &[(usize, T)], actual: &[(usize, T)], alg_name: String) -> Result<(), String> {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{alg_name}: Hit count mismatch: \nexp {expected:?}, \ngot {actual:?}",
    );

    for (i, (&(_, e), &(_, a))) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(e, a, "{alg_name}: Distance mismatch at index {i}: \nexp {expected:?}, \ngot {actual:?}",);
    }

    Ok(())
}
