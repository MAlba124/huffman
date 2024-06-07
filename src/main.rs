use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compress {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Expand {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Encode, Decode)]
enum NodeTyp {
    Value(u8),
    Link { left: Box<Node>, right: Box<Node> },
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Encode, Decode)]
struct Node {
    pub freq: usize,
    pub typ: NodeTyp,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Bit {
    Zero,
    One,
}

impl Bit {
    pub fn left() -> Self {
        Self::Zero
    }

    pub fn right() -> Self {
        Self::One
    }
}

impl Into<u8> for Bit {
    fn into(self) -> u8 {
        match self {
            Bit::Zero => 0,
            Bit::One => 1,
        }
    }
}

impl From<u8> for Bit {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::left(),
            1 => Self::right(),
            _ => panic!("Invalid"),
        }
    }
}

/// Find frequency of all the bytes in the input
fn create_frequency_map(data: &[u8]) -> HashMap<u8, usize> {
    let mut map: HashMap<u8, usize> = HashMap::new();
    for byte in data {
        let entry = map.entry(*byte).or_insert(0);
        *entry += 1;
    }
    map
}

/// Find the smallest frequency, remove it from the vec and return the node
fn min_freq(nodes: &mut Vec<Node>) -> Node {
    assert!(!nodes.is_empty());

    let mut min = usize::MAX;
    let mut min_index = 0;
    nodes.iter().enumerate().for_each(|(i, f)| {
        if f.freq < min {
            min = f.freq;
            min_index = i;
        }
    });

    let len = nodes.len();
    nodes.swap(min_index, len - 1);

    nodes.pop().unwrap()
}

/// Assemble the tree and return the root node
fn link_nodes(nodes: &[Node]) -> Node {
    assert!(!nodes.is_empty());

    let mut nodes = nodes.to_vec();
    while nodes.len() > 1 {
        let left = Box::new(min_freq(&mut nodes));
        let right = Box::new(min_freq(&mut nodes));
        nodes.push(Node {
            freq: left.freq + right.freq,
            typ: NodeTyp::Link { left, right },
        });
    }

    min_freq(&mut nodes)
}

fn create_huffman_tree(data: &[u8]) -> Node {
    let mut values: Vec<(u8, usize)> = create_frequency_map(data)
        .iter()
        .map(|(b, c)| (*b, *c))
        .collect();
    values.sort_by(|a, b| a.1.cmp(&b.1));

    let node_values: Vec<Node> = values
        .iter()
        // Discard the `frequency` value and create a Node value from the byte
        .map(|(v, f)| Node {
            freq: *f,
            typ: NodeTyp::Value(*v),
        })
        .collect();

    link_nodes(&node_values)
}

fn dfs(head: Node, byte: u8, code: &mut Vec<Bit>) -> bool {
    match head.typ {
        NodeTyp::Value(v) => {
            if v == byte {
                return true;
            }
            false
        }
        NodeTyp::Link { left, right } => {
            code.push(Bit::left());
            if dfs(*left, byte, code) {
                return true;
            } else {
                code.pop().unwrap();
            }

            code.push(Bit::right());
            if dfs(*right, byte, code) {
                return true;
            } else {
                code.pop().unwrap();
            }

            false
        }
    }
}
/// DFS the tree to find the value and return the bit path
fn search(tree: Node, byte: u8) -> Vec<Bit> {
    let mut bits: Vec<Bit> = Vec::new();
    if !dfs(tree, byte, &mut bits) {
        panic!("Not found");
    }

    bits
}

fn serialize_8_bits(bits: &[Bit; 8]) -> u8 {
    ((bits[0] as u8) << 7)
        + ((bits[1] as u8) << 6)
        + ((bits[2] as u8) << 5)
        + ((bits[3] as u8) << 4)
        + ((bits[4] as u8) << 3)
        + ((bits[5] as u8) << 2)
        + ((bits[6] as u8) << 1)
        + (bits[7] as u8)
}

fn compress_data(data: &[u8], tree: Node) -> Vec<u8> {
    let mut cache: HashMap<u8, Vec<Bit>> = HashMap::new();
    let mut bits = Vec::new();
    let mut bytes = Vec::new();
    for byte in data {
        if cache.contains_key(byte) {
            for bit in cache.get(byte).unwrap() {
                bits.push(*bit);
            }
        } else {
            let res = search(tree.clone(), *byte);
            let _ = cache.insert(*byte, res.clone());
            for bit in res {
                bits.push(bit);
            }
        }
    }

    let mut i = 0;
    loop {
        let mut eight_bits = [Bit::Zero; 8];
        let mut q = false;
        for j in 0..8 {
            if i >= bits.len() {
                q = true;
                break;
            }
            eight_bits[j] = bits[i];
            i += 1;
        }
        bytes.push(serialize_8_bits(&eight_bits));

        if q {
            break;
        }
    }

    bytes
}

/// Serialize the huffman tree and write the compressed data to a file
///
/// Binary file format:
/// First 8 bytes: length of uncompressed data (unsigned 64 bit little endian)
/// Next 8 bytes: length of huffman tree (TL) (unsigned 64 bit little endian)
/// data[16..(16+TL)]: serialized huffman tree
/// data[(16*2+TL)..EOF]: bitstream
fn write_compressed_file(
    f: &mut File,
    len: usize,
    tree: Node,
    compressed_data: &[u8],
) -> Result<(), std::io::Error> {
    let mut writer = BufWriter::new(f);

    let serialized_tree = bincode::encode_to_vec(&tree, bincode::config::standard()).unwrap();

    writer.write(&len.to_le_bytes())?;
    writer.write(&serialized_tree.len().to_le_bytes())?;
    writer.write(&serialized_tree)?;
    writer.write(compressed_data)?;

    Ok(())
}

fn bytes_to_bits(bytes: &[u8]) -> Vec<Bit> {
    let mut bits: Vec<Bit> = Vec::new();

    for byte in bytes {
        let mut mask = 0b10000000;
        for i in 0..8 {
            bits.push(Bit::from((byte & mask) >> (7 - i)));
            mask >>= 1;
        }
    }

    bits
}

/// Uncompress
fn expand(head: Node, bits: Vec<Bit>, len: usize) -> Vec<u8> {
    let mut curr = &head;
    let mut data: Vec<u8> = Vec::new();

    for bit in bits {
        if data.len() == len {
            break;
        }

        match bit {
            Bit::Zero => { // Left
                curr = match &curr.typ {
                    NodeTyp::Value(_) => panic!("Left: FAIL"),
                    NodeTyp::Link { left, .. } => &left,
                };

                if let NodeTyp::Value(v) = curr.typ {
                    data.push(v);
                    curr = &head;
                }
            }
            Bit::One => { // Right
                curr = match &curr.typ {
                    NodeTyp::Value(_) => panic!("Right: FAIL"),
                    NodeTyp::Link { right, .. } => &right,
                };

                if let NodeTyp::Value(v) = curr.typ {
                    data.push(v);
                    curr = &head;
                }
            }
        }
    }

    data
}

fn read_compressed_file(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);
    let mut eight_bytes = [0; 8];

    reader.read_exact(&mut eight_bytes)?;
    let len = usize::from_le_bytes(eight_bytes);
    reader.read_exact(&mut eight_bytes)?;
    let tree_len = usize::from_le_bytes(eight_bytes);
    let mut tree_data = Vec::new();
    for _ in 0..tree_len {
        let mut buf = [0; 1];
        reader.read_exact(&mut buf)?;
        tree_data.push(buf[0]);
    }

    let (tree, _tree_len): (Node, usize) =
        bincode::decode_from_slice(&tree_data, bincode::config::standard()).unwrap();

    let mut buf: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buf)?;
    let bits = bytes_to_bits(&buf);
    let data = expand(tree, bits, len);

    Ok(data)
}

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Compress { input, output } => {
            let mut input_f = File::open(input).unwrap();
            let mut output_f = File::create(output).unwrap();

            let mut data: Vec<u8> = Vec::new();
            input_f.read_to_end(&mut data).unwrap();

            let tree = create_huffman_tree(&data);
            let compressed_data = compress_data(&data, tree.clone());

            write_compressed_file(&mut output_f, data.len(), tree, &compressed_data).unwrap();
        }
        Commands::Expand { input, output } => {
            let mut output_f = File::create(output).unwrap();
            let expanded = read_compressed_file(&PathBuf::from(input)).unwrap();

            output_f.write_all(&expanded).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_bits() {
        assert_eq!(
            bytes_to_bits(&[0b10100001]),
            vec![
                Bit::One,
                Bit::Zero,
                Bit::One,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::One
            ]
        );
    }

    #[test]
    fn test_serialize_bits() {
        assert_eq!(
            serialize_8_bits(&[
                Bit::One,
                Bit::Zero,
                Bit::One,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::One
            ]),
            0b10100001
        );

        assert_eq!(
            serialize_8_bits(&[
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::One,
            ]),
            0b00000001
        );
    }
}
