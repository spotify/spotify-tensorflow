# spotify-tensorflow

## Raison d'être:

Provide Spotify specific Tensorflow helpers.

## Build

CI/CD is provided by [jenkins-scala](https://jenkins-scala.spotify.net/).

## Release 

spotify-tensorflow uses [pbr](http://docs.openstack.org/developer/pbr/) to manage versions.
Tag and push objects to master to release new version, to release:

```
git commit --allow-empty -m "Release x.y.z"
git tag x.y.z
git push --tags  origin master
```

Development versions are publised with `x.y.z.dev#` and available from `production` pip index. 
