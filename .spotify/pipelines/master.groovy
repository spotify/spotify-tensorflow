@Grab(group = 'com.spotify', module = 'pipeline-conventions', version = '1.0.7')

import static com.spotify.pipeline.Conventions.pipeline

pipeline(this) {
  notify.byMail(recipients: 'flatmap-squad+jenkins@spotify.com')

  group(name: 'PyPI Upload') {
    shell.run(cmd: 'sp-pypi-upload --fail-on-duplicate')
  }
}
