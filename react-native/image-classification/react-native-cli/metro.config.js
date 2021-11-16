const {getDefaultConfig} = require('metro-config');
module.exports = (async () => {
  const defaultConfig = await getDefaultConfig();
  const {assetExts, sourceExts} = defaultConfig.resolver;
  return {
    transformer: {
      getTransformOptions: async () => ({
        transform: {
          experimentalImportSupport: false,
          inlineRequires: false,
        },
      }),
    },
    resolver: {
      // Add bin to assetExts
      assetExts: [...assetExts, 'bin'],
      sourceExts: ['js', 'json', 'ts', 'tsx', 'jsx'],
    }
  };
})();
