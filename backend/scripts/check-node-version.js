const [major] = process.versions.node.split('.').map(Number);

if (major !== 24) {
  console.error('Unsupported Node.js version detected.');
  console.error(`Current version: ${process.versions.node}`);
  console.error('Required version: Node.js 24.x (portable runtime configured for this project).');
  console.error('Use: d:\\Major Project\\.portable-node\\node-v24.15.0-win-x64\\node.exe index.js');
  process.exit(1);
}
