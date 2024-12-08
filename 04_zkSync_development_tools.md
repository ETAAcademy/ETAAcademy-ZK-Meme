# ETAAcademy-ZKMeme: 4. ZkSync Development Tools

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>04. ZkSync Development Tools</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>zkSync_development_tools</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

More Details for ZKSync Development Tools: [English](https://docs.zksync.io/build/tooling/block-explorer/getting-started.html) [Chinese](https://github.com/WTFAcademy/WTF-zkSync/tree/main)

1.  **Blockchain Explorer:** [zkSync Era Block Explorer](https://explorer.zksync.io/);

2.  **zkSync Faucet** [Chainstack Faucet](https://faucet.chainstack.com/zksync-testnet-faucet);

3.  **zkSync CLI:** [zkSync CLI](https://github.com/matter-labs/zksync-cli) is a command line tool used to simplify zkSync development and interaction.

4.  Running with zksync-cli:

```
npx zksync-cli dev start;
```

5. Configure custom chains

```
npx zksync-cli config chains;
```

6.  Used to operate contracts on the chain (read, write, etc.)

```
npx zksync-cli contract read
```

7. Query on-chain transaction information (transaction status, transfer amount, gas fee, etc.);

```
npx zksync-cli transaction info [options]

```

```
options:
-full: query detailed information
-raw: display the original JSON response;
```

8.  Create projects (front-end, smart contracts and scripts).

```
npx zksync-cli create
```

9.  Wallet is used to manage wallet-related functions (transfers, balance inquiries, etc.). Balance query:

```
npx zksync-cli wallet balance [options]
```

```
options:
-address: The address where you want to query the balance;
-token: If you want to query ERC20 tokens, you can add this parameter and pass in the contract address; -chain: Specify which chain to query on;
-rpc: specify the RPC URL; used to handle cross-chain operations between Ethernet and zkSync.
```

10. Use `deposit` to transfer assets `from L1 to L2`:

```
    npx zksync-cli bridge deposit;
```

11. Use `withdraw` to transfer assets `from L2 to L1`:

```
npx zksync-cli bridge withdraw`
```

12. [Remix IDE](https://remix.ethereum.org/) also supports zkSync contract development(you need to start docker) by the [zkSync Era Remix Plugin](https://medium.com/nethermind-eth/the-zksync-era-remix-plugin-a-how-to-guide-fc54e8d24bd3):

```
npx zksync-cli dev start
```

13. Use `zksync-cli` to quickly create Hardhat projects:

```
npx zksync-cli create
```

14. [zksync-ethers](https://github.com/zksync-sdk/zksync-ethers) extends the `ethers` library to support zkSync-specific features (such as account abstraction)

15. [foundry-zksync](https://github.com/matter-labs/foundry-zksync) allows users to use foundry to develop smart contracts on zkSync, introducing `zkforge` and `zkcast` to extend the original ` forge` and `cast` make it easier for developers to develop in zkSync.

16. The command can create a template test project named `test-demo`
    Run the following two lines of commands to install the third-party libraries that the project depends on, and add an additional zkSync data test tool package

```
npx zksync-cli create test-demo

```

```
npm install
npm install -D @matterlabs/hardhat-zksync-chai-matchers @nomicfoundation/hardhat-chai-matchers @nomiclabs/hardhat-ethers

```

Finally, run `npm test` to start the test. All test files in the `test` directory will be run. (`zksolc` and `solc` will be downloaded when starting for the first time. )
