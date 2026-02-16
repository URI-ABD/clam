//! Tests for `PanCakes`.

#![expect(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use core::fmt::Debug;

use rand::prelude::*;

use abd_clam::{
    DistanceValue, NamedAlgorithm, Tree, cakes,
    pancakes::{Codec, Compressible, CompressiveSearch, MaybeCompressed, PancakesTree},
};

#[derive(Debug, PartialEq, Eq, Clone)]
struct TestItem<const N: usize> {
    arr: [char; N],
}

impl<const N: usize> Compressible for TestItem<N> {
    // A simple compression scheme that encodes each pairwise difference between characters along with the index.
    type Compressed = Box<[(u8, char)]>;

    fn original_size(&self) -> usize {
        core::mem::size_of_val(self)
    }

    fn compressed_size(compressed: &Self::Compressed) -> usize {
        core::mem::size_of_val(compressed)
    }
}

#[derive(Debug, Clone)]
struct SmallItemCodec<const N: usize>;

impl<const N: usize> Codec<TestItem<N>> for SmallItemCodec<N> {
    fn compress(&self, reference: &TestItem<N>, target: &TestItem<N>) -> Box<[(u8, char)]> {
        reference
            .arr
            .iter()
            .zip(target.arr.iter())
            .enumerate()
            .filter_map(|(i, (&a, &b))| if a == b { None } else { Some((i as u8, b)) })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn decompress(&self, reference: &TestItem<N>, compressed: &Box<[(u8, char)]>) -> TestItem<N> {
        let mut arr = reference.arr;
        for &(i, c) in compressed {
            arr[i as usize] = c;
        }
        TestItem { arr }
    }
}

#[test]
fn pair() {
    type Tenner = TestItem<10>;

    let codec = SmallItemCodec::<10>;
    let item1 = Tenner { arr: ['a'; 10] };
    let item2 = Tenner {
        arr: ['a', 'c', 'a', 't', 'a', 'g', 'a', 'a', 'a', 'a'],
    };

    assert_eq!(item1.original_size(), 10 * std::mem::size_of::<char>());
    assert_eq!(item2.original_size(), 10 * std::mem::size_of::<char>());

    let compressed = codec.compress(&item1, &item2);
    assert_eq!(compressed, vec![(1, 'c'), (3, 't'), (5, 'g')].into_boxed_slice());

    let original_size = item2.original_size();
    let compressed_size = Tenner::compressed_size(&compressed);
    assert_eq!(compressed_size, 16);
    assert!(
        compressed_size < original_size,
        "Compression should reduce size. Got compressed size {compressed_size} and original size {original_size}."
    );

    let decompressed = codec.decompress(&item1, &compressed);
    assert_eq!(item2, decompressed);
    assert_eq!(item2.original_size(), decompressed.original_size());
}

fn gen_test_item<R: Rng, const N: usize>(rng: &mut R, chars: &[char]) -> Result<TestItem<N>, String> {
    let mut arr = [chars[0]; N];
    for c in &mut arr {
        *c = *chars.choose(rng).ok_or_else(|| "chars should not be empty".to_string())?;
    }
    Ok(TestItem { arr })
}

fn hamming<const N: usize>(a: &TestItem<N>, b: &TestItem<N>) -> usize {
    a.arr.iter().zip(b.arr.iter()).filter(|(a, b)| a != b).count()
}

fn compare_trees<Id, I, T, A, M, C>(tree: &Tree<Id, I, T, A, M>, compressed_tree: &PancakesTree<Id, I, T, A, M, C>, ratio: f64)
where
    Id: Eq + Debug,
    I: Compressible + Eq + Debug,
    T: Debug,
    A: Debug,
    C: Codec<I>,
{
    assert_eq!(tree.cardinality(), compressed_tree.cardinality(), "Trees should have the same cardinality.");

    let (compressed, uncompressed): (Vec<_>, Vec<_>) = compressed_tree
        .iter_items()
        .enumerate()
        .map(|(i, (id, item, _))| (i, id, item))
        .partition(|(_, _, item)| matches!(item, MaybeCompressed::Compressed(_)));
    let compressed = compressed.into_iter().map(|(i, id, _)| (i, id)).collect::<Vec<_>>();
    let uncompressed = uncompressed.into_iter().map(|(i, id, _)| (i, id)).collect::<Vec<_>>();
    assert_eq!(
        uncompressed.len(),
        1,
        "Only the root center should be uncompressed. Num uncompressed items: {}.\nCompressed item IDs: {compressed:?}.\nUncompressed item IDs: {uncompressed:?}",
        uncompressed.len()
    );

    let items_size = tree.iter_items().map(|(_, item, _)| item.original_size()).sum::<usize>();
    let compressed_size = compressed_tree.iter_items().map(|(_, item, _)| item.size()).sum::<usize>();

    assert!(
        compressed_size < items_size,
        "Compression should reduce size of items. Got compressed size {compressed_size} and original size {items_size}."
    );
    let items_ratio = compressed_size as f64 / items_size as f64;
    assert!(
        items_ratio < ratio,
        "Items not compressed enough. Got compressed size {compressed_size} and original size {items_size}, for a ratio of {items_ratio:.2}."
    );
}

#[test]
fn compression() -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(42);
    let chars = ['a', 'c', 't', 'g'];
    let items = (0_usize..1_000)
        .map(|_| gen_test_item::<_, 6>(&mut rng, &chars))
        .collect::<Result<Vec<_>, _>>()?;

    let tree = Tree::new_minimal(items, hamming)?;
    let compressed_tree = tree.clone().compress(SmallItemCodec, 3);
    compare_trees(&tree, &compressed_tree, 0.68);

    let decompressed_tree = compressed_tree.decompress().0;
    assert_eq!(
        tree.cardinality(),
        decompressed_tree.cardinality(),
        "Decompressed tree should have the same number of items as the original tree."
    );

    for (i, ((l_id, l_item, _), (r_id, r_item, _))) in tree.iter_items().zip(decompressed_tree.iter_items()).enumerate() {
        assert_eq!(l_id, r_id, "Item IDs should match after decompression. at index {i}");
        assert_eq!(l_item, r_item, "Items should match after decompression at index {i}.");
    }

    Ok(())
}

#[test]
fn par_compression() -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(42);
    let chars = ['a', 'c', 't', 'g'];
    let items = (0_usize..20_000)
        .map(|_| gen_test_item::<_, 8>(&mut rng, &chars))
        .collect::<Result<Vec<_>, _>>()?;

    let tree = Tree::par_new_minimal(items, hamming)?;
    let compressed_tree = tree.clone().par_compress(SmallItemCodec, 3);
    compare_trees(&tree, &compressed_tree, 0.51);

    let decompressed_tree = compressed_tree.par_decompress().0;
    assert_eq!(
        tree.cardinality(),
        decompressed_tree.cardinality(),
        "Decompressed tree should have the same number of items as the original tree."
    );

    for (i, ((l_id, l_item, _), (r_id, r_item, _))) in tree.iter_items().zip(decompressed_tree.iter_items()).enumerate() {
        assert_eq!(l_id, r_id, "Item IDs should match after decompression. at index {i}");
        assert_eq!(l_item, r_item, "Items should match after decompression at index {i}.");
    }

    Ok(())
}

#[test]
fn search() -> Result<(), String> {
    let mut rng = rand::rng();
    let chars = ['a', 'c', 't', 'g'];
    let data = (0..1_000).map(|_| gen_test_item::<_, 6>(&mut rng, &chars)).collect::<Result<Vec<_>, _>>()?;
    let query = gen_test_item::<_, 6>(&mut rng, &chars)?;

    let mut tree = Tree::new_minimal(data, hamming)?.compress(SmallItemCodec, 3);

    for radius in [1, 2, 4] {
        let linear_alg = cakes::RnnLinear::new(radius);
        let linear_hits = linear_alg.compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        tree.compress_from_root();

        let chess_alg = cakes::RnnChess::new(radius);
        let chess_hits = chess_alg.compressive_search(&mut tree, &query);
        let chess_hits = sort_nondescending(chess_hits);
        tree.compress_from_root();

        check_hits(&linear_hits, &chess_hits, &chess_alg);
    }

    for radius in [1, 2, 4] {
        let linear_alg = cakes::RnnLinear::new(radius);
        let linear_hits = linear_alg.compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);

        let chess_alg = cakes::RnnChess::new(radius);
        let chess_hits = chess_alg.compressive_search(&mut tree, &query);
        let chess_hits = sort_nondescending(chess_hits);

        check_hits(&linear_hits, &chess_hits, &chess_alg);
    }
    tree.compress_from_root();

    for k in [1, 10, 20] {
        let linear_alg = cakes::KnnLinear::new(k);
        let linear_hits = linear_alg.compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(tree.cardinality()),
            "Not enough linear hits {} for k={}",
            linear_hits.len(),
            k.min(tree.cardinality())
        );
        tree.compress_from_root();

        let dfs_alg = cakes::KnnDfs::new(k);
        let dfs_hits = dfs_alg.compressive_search(&mut tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        tree.compress_from_root();
        check_hits(&linear_hits, &dfs_hits, &dfs_alg);

        let bfs_alg = cakes::KnnBfs::new(k);
        let bfs_hits = bfs_alg.compressive_search(&mut tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        tree.compress_from_root();
        check_hits(&linear_hits, &bfs_hits, &bfs_alg);

        let rrnn_alg = cakes::KnnRrnn::new(k);
        let rrnn_hits = rrnn_alg.compressive_search(&mut tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        tree.compress_from_root();
        check_hits(&linear_hits, &rrnn_hits, &rrnn_alg);
    }

    for k in [1, 10, 20] {
        let linear_alg = cakes::KnnLinear::new(k);
        let linear_hits = linear_alg.compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(tree.cardinality()),
            "Not enough linear hits {} for k={}",
            linear_hits.len(),
            k.min(tree.cardinality())
        );

        let dfs_alg = cakes::KnnDfs::new(k);
        let dfs_hits = dfs_alg.compressive_search(&mut tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&linear_hits, &dfs_hits, &dfs_alg);

        let bfs_alg = cakes::KnnBfs::new(k);
        let bfs_hits = bfs_alg.compressive_search(&mut tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&linear_hits, &bfs_hits, &bfs_alg);

        let rrnn_alg = cakes::KnnRrnn::new(k);
        let rrnn_hits = rrnn_alg.compressive_search(&mut tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&linear_hits, &rrnn_hits, &rrnn_alg);
    }
    tree.compress_from_root();

    Ok(())
}

#[test]
fn par_search() -> Result<(), String> {
    let mut rng = rand::rng();
    let chars = ['a', 'c', 't', 'g'];
    let data = (0..4_000).map(|_| gen_test_item::<_, 8>(&mut rng, &chars)).collect::<Result<Vec<_>, _>>()?;
    let query = gen_test_item::<_, 8>(&mut rng, &chars)?;

    let mut tree = Tree::new_minimal(data, hamming)?.par_compress(SmallItemCodec, 3);

    for radius in [1, 2, 4] {
        let linear_alg = cakes::RnnLinear::new(radius);
        let linear_hits = linear_alg.par_compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        tree.par_compress_from_root();

        let chess_alg = cakes::RnnChess::new(radius);
        let chess_hits = chess_alg.par_compressive_search(&mut tree, &query);
        let chess_hits = sort_nondescending(chess_hits);
        tree.par_compress_from_root();

        check_hits(&linear_hits, &chess_hits, &chess_alg);
    }

    for radius in [1, 2, 4] {
        let linear_alg = cakes::RnnLinear::new(radius);
        let linear_hits = linear_alg.par_compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);

        let chess_alg = cakes::RnnChess::new(radius);
        let chess_hits = chess_alg.par_compressive_search(&mut tree, &query);
        let chess_hits = sort_nondescending(chess_hits);

        check_hits(&linear_hits, &chess_hits, &chess_alg);
    }
    tree.par_compress_from_root();

    for k in [1, 10, 20] {
        let linear_alg = cakes::KnnLinear::new(k);
        let linear_hits = linear_alg.par_compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(tree.cardinality()),
            "Not enough linear hits {} for k={}",
            linear_hits.len(),
            k.min(tree.cardinality())
        );
        tree.par_compress_from_root();

        let dfs_alg = cakes::KnnDfs::new(k);
        let dfs_hits = dfs_alg.par_compressive_search(&mut tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        tree.par_compress_from_root();
        check_hits(&linear_hits, &dfs_hits, &dfs_alg);

        let bfs_alg = cakes::KnnBfs::new(k);
        let bfs_hits = bfs_alg.par_compressive_search(&mut tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        tree.par_compress_from_root();
        check_hits(&linear_hits, &bfs_hits, &bfs_alg);

        let rrnn_alg = cakes::KnnRrnn::new(k);
        let rrnn_hits = rrnn_alg.par_compressive_search(&mut tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        tree.par_compress_from_root();
        check_hits(&linear_hits, &rrnn_hits, &rrnn_alg);
    }

    for k in [1, 10, 20] {
        let linear_alg = cakes::KnnLinear::new(k);
        let linear_hits = linear_alg.par_compressive_search(&mut tree, &query);
        let linear_hits = sort_nondescending(linear_hits);
        assert_eq!(
            linear_hits.len(),
            k.min(tree.cardinality()),
            "Not enough linear hits {} for k={}",
            linear_hits.len(),
            k.min(tree.cardinality())
        );

        let dfs_alg = cakes::KnnDfs::new(k);
        let dfs_hits = dfs_alg.par_compressive_search(&mut tree, &query);
        let dfs_hits = sort_nondescending(dfs_hits);
        check_hits(&linear_hits, &dfs_hits, &dfs_alg);

        let bfs_alg = cakes::KnnBfs::new(k);
        let bfs_hits = bfs_alg.par_compressive_search(&mut tree, &query);
        let bfs_hits = sort_nondescending(bfs_hits);
        check_hits(&linear_hits, &bfs_hits, &bfs_alg);

        let rrnn_alg = cakes::KnnRrnn::new(k);
        let rrnn_hits = rrnn_alg.par_compressive_search(&mut tree, &query);
        let rrnn_hits = sort_nondescending(rrnn_hits);
        check_hits(&linear_hits, &rrnn_hits, &rrnn_alg);
    }
    tree.par_compress_from_root();

    Ok(())
}

fn sort_nondescending(mut items: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    items.sort_by_key(|(_, d)| *d);
    items
}

fn check_hits<T: DistanceValue, Alg: NamedAlgorithm>(expected: &[(usize, T)], actual: &[(usize, T)], alg: &Alg) {
    assert_eq!(expected.len(), actual.len(), "{alg}: Hit count mismatch: \nexp {expected:?}, \ngot {actual:?}",);

    for (i, (&(_, e), &(_, a))) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(e, a, "{alg}: Distance mismatch at index {i}: \nexp {expected:?}, \ngot {actual:?}",);
    }
}
