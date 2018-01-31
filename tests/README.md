## Generating test resources

```
cd tests/resources/tf-test-resource
sbt "runMain com.spotify.GenerateTFRecords --runner=DirectRunner --output=./tf-records"
```