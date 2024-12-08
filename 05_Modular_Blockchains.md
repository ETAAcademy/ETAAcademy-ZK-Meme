# ETAAcademy-ZKMeme: 5. Monolithic and Modular Blockchains

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>05. Monolithic and Modular Blockchains</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>monolithic_modular_blockchains</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

Traditional blockchains, like Bitcoin and Solana, handle consensus (data is verified as authentic), execution (blockchain nodes process transactions,), data availability (block producers must publish the data of each block for others to download and store) and settlement (guaranteeing that transactions committed to chain history are irreversible or "immutable") within the same layer, known as monolithic blockchains. Solana prioritizes scalability over decentralization and security, achieving high transaction speeds but requiring high hardware requirements.

In contrast, modular blockchains focus on specific tasks and outsource others to separate layers. Ethereum, transitioning towards a modular framework, implements sharding to split the blockchain into sub-chains and rollups to handle transactions while outsourcing consensus, data availability, and settlement to the parent chain.

Rollup designs, such as optimistic rollups and zero-knowledge rollups, adopt a modular approach by executing transactions on separate chains while relying on Ethereum for data availability, consensus, and settlement.

Celestia and Polygon Avail are emerging modular blockchains focused on consensus and data availability, serving as potential data availability layers for other execution layers like rollups.
