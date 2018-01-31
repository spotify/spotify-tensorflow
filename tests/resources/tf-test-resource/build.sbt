import sbt._
import Keys._

val scioVersion = "0.5.0-alpha1"
val beamVersion = "2.2.0"
val scalaMacrosVersion = "2.1.0"
val featranVersion = "0.1.16"

lazy val commonSettings = Defaults.coreDefaultSettings ++ Seq(
  organization          := "spotify",
  // Semantic versioning http://semver.org/
  version               := "0.1.0-SNAPSHOT",
  scalaVersion          := "2.11.12",
  scalacOptions         ++= Seq("-target:jvm-1.8",
                                "-deprecation",
                                "-feature",
                                "-unchecked"),
  javacOptions          ++= Seq("-source", "1.8",
                                "-target", "1.8")
)

lazy val paradiseDependency =
  "org.scalamacros" % "paradise" % scalaMacrosVersion cross CrossVersion.full
lazy val macroSettings = Seq(
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  addCompilerPlugin(paradiseDependency)
)

lazy val noPublishSettings = Seq(
  publish := {},
  publishLocal := {},
  publishArtifact := false
)

lazy val root: Project = Project(
  "tf-test-resource",
  file(".")
).settings(
  commonSettings ++ macroSettings ++ noPublishSettings,
  description := "tf-test-resource",
  libraryDependencies ++= Seq(
    "com.spotify" %% "scio-core" % scioVersion,
    "com.spotify" %% "scio-tensorflow" % scioVersion,
    "com.spotify" %% "scio-test" % scioVersion % "test",
    "com.spotify" %% "featran-core" % featranVersion,
    "com.spotify" %% "featran-scio" % featranVersion,
    "com.spotify" %% "featran-tensorflow" % featranVersion,
     "org.apache.beam" % "beam-runners-direct-java" % beamVersion,
    "org.slf4j" % "slf4j-simple" % "1.7.25"
  )
).enablePlugins(PackPlugin)

lazy val repl: Project = Project(
  "repl",
  file(".repl")
).settings(
  commonSettings ++ macroSettings ++ noPublishSettings,
  description := "Scio REPL for tf-test-resource",
  libraryDependencies ++= Seq(
    "com.spotify" %% "scio-repl" % scioVersion
  ),
  mainClass in Compile := Some("com.spotify.scio.repl.ScioShell")
).dependsOn(
  root
)
