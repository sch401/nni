// Copyright (c) Microsoft Corporation. Licensed under the MIT license.
'use strict';
const fs = require('fs');
const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const paths = require('./paths');
const getClientEnvironment = require('./env');
module.exports = function(mode) {
  const production = mode === 'production';
  return {
    mode: production ? 'production' : 'development',
    entry: paths.appIndexJs,
    output: {path: paths.appBuild, filename: 'static/js/[name].[contenthash:8].js',
      chunkFilename: 'static/js/[name].[contenthash:8].chunk.js', publicPath: './'},
    resolve: {extensions: ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.json'], alias: {
      'monaco-editor/esm/vs': path.join(paths.appPath, 'node_modules/monaco-editor/esm/vs'),
      '@': paths.appSrc, '@components': path.join(paths.appSrc, 'components'),
      '@static': path.join(paths.appSrc, 'static'), '@style': path.join(paths.appSrc, 'static/style'),
      '@model': path.join(paths.appSrc, 'static/model')}},
    module: {rules: [
      {test: /\.[jt]sx?$/, include: paths.appSrc, use: {loader: 'babel-loader', options: {
        babelrc: false, configFile: false, presets: [['@babel/preset-react', {runtime: 'automatic'}],
          ['@babel/preset-typescript', {onlyRemoveTypeImports: false}]]}}},
      {test: /\.css$/, use: [MiniCssExtractPlugin.loader, 'css-loader']},
      {test: /\.s[ac]ss$/, use: [MiniCssExtractPlugin.loader, 'css-loader', 'sass-loader']},
      {test: /\.(png|jpg|jpeg|gif|svg|woff2?|ttf|eot)$/, type: 'asset/resource'}]},
    plugins: [
      new HtmlWebpackPlugin({templateContent: fs.readFileSync(paths.appHtml, 'utf8').replaceAll('%PUBLIC_URL%', '.')}),
      new MiniCssExtractPlugin({filename: 'static/css/[name].[contenthash:8].css'}),
      new MonacoWebpackPlugin({languages: ['json'], filename: 'static/js/[name].worker.js'}),
      new webpack.DefinePlugin(getClientEnvironment('.').stringified)],
    devtool: production ? false : 'source-map'
  };
};
